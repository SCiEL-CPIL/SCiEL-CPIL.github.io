"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """Construct the target multiplication tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    """
    Verify if U, V, W decompose T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _try_rational_snapping(U, V, W, T, max_denom=4):
    """
    Try to snap the float solution to rational numbers with small denominators.
    Returns True if a snapped version verifies exactly, and updates U, V, W in place.
    """
    candidates = [
        (1, 1.0),
        (2, 0.5),
        (4, 0.25)
    ]
    
    for denom, step in candidates:
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-9):
            U[:] = U_snap
            V[:] = V_snap
            W[:] = W_snap
            return True
            
    return False

def _als_polish(U, V, W, T, n, m, p, iterations=10, tol=1e-10):
    """
    Polish the solution using Alternating Least Squares.
    Uses vectorized operations for speed.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    prev_loss = np.inf
    
    for it in range(iterations):
        # Solve for U: min || T.reshape(nm, mp*pn) - U @ (V⊗W).T ||_F
        # VW_mat[b*pn+c, :] = V[b, :] * W[c, :]
        VW_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                VW_mat[b*pn+c, :] = V[b, :] * W[c, :]
        
        T_mat = T.reshape(nm, mp*pn)
        try:
            # Use least squares for better conditioning
            U_new, _, _, _ = np.linalg.lstsq(VW_mat, T_mat.T, rcond=None)
            U_new = U_new.T
        except np.linalg.LinAlgError:
            U_new = U
        
        # Solve for V
        UW_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                UW_mat[a*pn+c, :] = U_new[a, :] * W[c, :]
        
        T_mat_v = T.reshape(mp, nm*pn)
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW_mat, T_mat_v.T, rcond=None)
            V_new = V_new.T
        except np.linalg.LinAlgError:
            V_new = V
        
        # Solve for W
        UV_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UV_mat[a*mp+b, :] = U_new[a, :] * V_new[b, :]
        
        T_mat_w = T.reshape(pn, nm*mp)
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV_mat, T_mat_w.T, rcond=None)
            W_new = W_new.T
        except np.linalg.LinAlgError:
            W_new = W
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        loss = np.sum((T - recon)**2)
        
        U, V, W = U_new, V_new, W_new
        
        if abs(prev_loss - loss) < tol and loss < 1e-8:
            break
        prev_loss = loss

    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """
    Generate structured initializations exploiting the matrix multiplication tensor structure.
    Uses the fact that T[i*m+j, j*p+k, k*n+i] = 1 to create informed initializations.
    """
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy 1: Random but with sparsity pattern matching the tensor support
    # For each (i,j,k) in the tensor, the entry (i*m+j, j*p+k, k*n+i) is 1.
    # We can initialize each rank-1 component to "aim" at a subset of these entries.
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Assign each (i,j,k) triple to a rank component
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total_triples = len(triples)
    
    # Distribute triples among rank components
    if rank <= total_triples:
        # Each component handles ~total_triples/rank triples
        for r in range(rank):
            # Pick a random subset of triples for this component
            start = int(r * total_triples / rank)
            end = int((r + 1) * total_triples / rank)
            if r == rank - 1:
                end = total_triples
            
            for idx in range(start, end):
                i, j, k = triples[idx]
                # Add contribution to this rank component
                U[i*m+j, r] += 1.0 / np.sqrt(end - start)
                V[j*p+k, r] += 1.0 / np.sqrt(end - start)
                W[k*n+i, r] += 1.0 / np.sqrt(end - start)
    
    # Add small random perturbation to break symmetry
    U += np.random.randn(nm, rank) * 0.05
    V += np.random.randn(mp, rank) * 0.05
    W += np.random.randn(pn, rank) * 0.05
    
    return U, V, W

def _kronecker_init(n, m, p, rank, seed):
    """
    Use Kronecker product structure for initialization.
    Decompose 2x5x6 as combinations of smaller matrix multiplications.
    """
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # For 2x5x6, we can think of it as:
    # - 2x5 matrix times 5x6 matrix
    # The tensor has entries T[i*5+j, j*6+k, k*2+i] = 1
    
    # Initialize using a block structure
    # Split the rank into blocks corresponding to different (i,j) or (j,k) pairs
    
    # Method: use the fact that for fixed i, the sub-tensor involves m*p entries
    # For each i (0..n-1), we have a m x p x (p*n) structure
    
    # Simple approach: initialize each column to target a random (i,j,k)
    for r in range(rank):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        U[i*m+j, r] = 1.0
        V[j*p+k, r] = 1.0
        W[k*n+i, r] = 1.0
    
    # Add random components
    U += np.random.randn(nm, rank) * 0.1
    V += np.random.randn(mp, rank) * 0.1
    W += np.random.randn(pn, rank) * 0.1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (Rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: evaluator kills at 150s
    
    # Try ranks from 20 to 59, with more restarts at lower ranks
    for rank in range(20, best_rank):
        if time.time() > deadline - 25:
            break
        
        # More restarts for lower ranks (more constrained, need more tries)
        max_restarts = max(10, 40 - rank)
        
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Choose initialization strategy based on restart number
            if restart % 3 == 0:
                U, V, W = _structured_init(n, m, p, rank, seed=42 + restart * 1000)
            elif restart % 3 == 1:
                U, V, W = _kronecker_init(n, m, p, rank, seed=84 + restart * 1000)
            else:
                U = np.random.randn(nm, rank) * 0.1
                V = np.random.randn(mp, rank) * 0.1
                W = np.random.randn(pn, rank) * 0.1
            
            # Adam Optimizer with adaptive learning rate
            v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            beta1, beta2 = 0.9, 0.999
            lr = 0.02  # Slightly higher initial LR
            t = 0
            best_loss = np.inf
            best_U_copy = U.copy()
            best_V_copy = V.copy()
            best_W_copy = W.copy()
            
            for step in range(400):
                if time.time() > deadline - 5:
                    break
                    
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - recon
                loss = np.sum(R**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U_copy = U.copy()
                    best_V_copy = V.copy()
                    best_W_copy = W.copy()
                    if loss < 1e-10:
                        break
                        
                # Gradients
                gU = -np.einsum('abc,br,cr->ar', R, V, W)
                gV = -np.einsum('abc,ar,cr->br', R, U, W)
                gW = -np.einsum('abc,ar,br->cr', R, U, V)
                
                grads = [gU, gV, gW]
                params = [U, V, W]
                
                t += 1
                for i in range(3):
                    m_[i] = beta1 * m_[i] + (1 - beta1) * grads[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
                    m_hat = m_[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    
                U = params[0]
                V = params[1]
                W = params[2]
                
                # Adaptive LR: decay every 100 steps, but not too aggressively
                if step % 100 == 0 and step > 0:
                    lr *= 0.7
                
                # Check for convergence
                if step % 20 == 0 and loss < 1e-6:
                    if _verify_fast(U, V, W, T, tol=1e-8):
                        found = True
                        break
                        
            if not found and best_loss < 1e-4:
                # Use the best solution found as a starting point for ALS polish
                U, V, W = best_U_copy, best_V_copy, best_W_copy
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=8, tol=1e-12)
                if _verify_fast(U, V, W, T, tol=1e-8):
                    found = True
            
            if found:
                if _verify_fast(U, V, W, T, tol=1e-8):
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    _try_rational_snapping(U, V, W, T)
                    break

        if found:
            break

    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
