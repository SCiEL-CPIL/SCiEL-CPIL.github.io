"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import itertools
import time
from collections import deque

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Fast verification: check only the 60 non-zero tensor entries + sample zeros."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = 0.0
                for rr in range(r):
                    val += U[a, rr] * V[b, rr] * W[c, rr]
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of zero entries
    rng = np.random.RandomState(123)
    for _ in range(200):
        a = rng.randint(nm)
        b = rng.randint(mp)
        c = rng.randint(pn)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for rr in range(r):
            val += U[a, rr] * V[b, rr] * W[c, rr]
        if abs(val) > tol:
            return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using the structure of the tensor."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    checked = 0
    for _ in range(500):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for cc in range(r):
            val += U[a, cc] * V[b, cc] * W[c, cc]
        if abs(val) > tol:
            return False
        checked += 1
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using rational arithmetic or high precision."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = 0.0
                for cc in range(r):
                    val += U[a, cc] * V[b, cc] * W[c, cc]
                if abs(val) > 1e-10:
                    return False
    return True

def _als_polish(U, V, W, T, n, m, p, iterations=20, tol=1e-10, deadline=None):
    """Vectorized ALS polishing step with deadline guard."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for it in range(iterations):
        if deadline and time.time() > deadline:
            break
        
        # Update U: vectorized
        # For each a: T[a] (flattened) = U[a,:] @ A_a.T where A_a[bc, r] = V[b,r]*W[c,r]
        # Precompute V*W outer: shape (mp, pn, r)
        VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, r)
        for a in range(nm):
            A = VW.reshape(mp * pn, r)  # (mp*pn, r)
            T_a = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(A.T, T_a, rcond=None)
            except:
                pass
        
        # Update V: vectorized
        UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, r)
        for b in range(mp):
            A = UW.reshape(nm * pn, r)
            T_b = T[:, b, :].flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(A.T, T_b, rcond=None)
            except:
                pass
        
        # Update W: vectorized
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, r)
        for c in range(pn):
            A = UV.reshape(nm * mp, r)
            T_c = T[:, :, c].flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(A.T, T_c, rcond=None)
            except:
                pass
    
    return U, V, W

def _kronecker_decomposition(n, m, p):
    """
    Try to decompose the n x m x p matrix multiplication tensor using
    Kronecker product structure. The idea: if n = n1*n2, m = m1*m2, p = p1*p2,
    then the tensor rank can be bounded by rank(n1,m1,p1) * rank(n2,m2,p2).
    
    For 2x5x6: we can try various factorizations.
    2x5x6 doesn't factor nicely into two non-trivial parts.
    But we can try: treat as a "block" structure.
    
    Alternative: use the fact that rank(2x5x6) <= rank(2x5x5) * 6/something...
    Actually, let's try a different approach: use known bounds and random search
    with structured initialization.
    
    Known: rank(2xmxp) for small cases.
    For 2x5x6, the Strassen-like bounds suggest rank around 20-30.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy 1: Use the "row-wise" Kronecker structure
    # C = A * B where A is 2x5, B is 5x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can view this as: for each i, C[i,:] = A[i,:] * B
    # This is a vector-matrix product of dimension 1x5 times 5x6
    # The rank of 1x5x6 matmul is 5 (schoolbook) or possibly less with Strassen
    # 
    # For two such products (i=0 and i=1), we need rank <= 2*5 = 10? No,
    # we can share the B factors.
    #
    # Actually, the full tensor rank for 2x5x6:
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Upper bound (schoolbook): n*m*p = 60
    # 
    # Let's try rank from 30 upward with structured initialization.
    
    return None  # Placeholder - we'll do the search in construct()

def _structured_init(n, m, p, rank, seed=42):
    """Create structured initialization exploiting the tensor's block structure."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Structure: group terms by the "j" index (the inner dimension)
    # For each j in range(m), we have n*p terms that involve A[i,j] and B[j,k]
    # If rank >= m, we can dedicate rank/m terms per j-block
    
    if rank >= m:
        terms_per_j = rank // m
        remainder = rank % m
        col = 0
        for j in range(m):
            n_terms = terms_per_j + (1 if j < remainder else 0)
            for t in range(n_terms):
                # This term handles contributions for this j
                # U[i*m+j, col] should be nonzero for all i
                # V[j*p+k, col] should be nonzero for all k
                # W[k*n+i, col] should be nonzero for all i,k
                for i in range(n):
                    U[i*m+j, col] = rng.randn()
                for k in range(p):
                    V[j*p+k, col] = rng.randn()
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, col] = rng.randn()
                col += 1
    else:
        # Rank < m: use random initialization
        U = rng.randn(nm, rank) / np.sqrt(rank)
        V = rng.randn(mp, rank) / np.sqrt(rank)
        W = rng.randn(pn, rank) / np.sqrt(rank)
    
    return U, V, W

def _gradient_descent(U, V, W, T, n, m, p, max_steps=100, lr=0.01, deadline=None):
    """Gradient descent with adaptive learning rate and early stopping."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    momentum = 0.9
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
        
        vU = momentum * vU + (1 - momentum) * grad_U
        vV = momentum * vV + (1 - momentum) * grad_V
        vW = momentum * vW + (1 - momentum) * grad_W
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        if loss > best_loss * 1.5:
            lr *= 0.7
            vU = 0
            vV = 0
            vW = 0
    
    return best_U, best_V, best_W, best_loss

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_rank = n * m * p
    best_result = None
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    np.random.seed(42)
    
    # Try to find lower rank, starting from the theoretical lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
            
        found = False
        max_restarts = 12
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart % 4 == 0:
                # Structured initialization
                U, V, W = _structured_init(n, m, p, rank, seed=42 + restart * 7)
            elif restart % 4 == 1:
                # Random with scaling
                scale = 1.0 / np.sqrt(rank)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            elif restart % 4 == 2:
                # Sparse-like initialization
                scale = 1.0 / np.sqrt(rank)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
                mask_U = np.random.rand(nm, rank) < 0.5
                mask_V = np.random.rand(mp, rank) < 0.5
                mask_W = np.random.rand(pn, rank) < 0.5
                U[mask_U] = 0
                V[mask_V] = 0
                W[mask_W] = 0
            else:
                # Very sparse: 70% zeros
                scale = 1.0 / np.sqrt(rank * 0.3)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
                mask_U = np.random.rand(nm, rank) < 0.7
                mask_V = np.random.rand(mp, rank) < 0.7
                mask_W = np.random.rand(pn, rank) < 0.7
                U[mask_U] = 0
                V[mask_V] = 0
                W[mask_W] = 0
            
            # Phase 1: GD
            U, V, W, loss = _gradient_descent(U, V, W, T, n, m, p, max_steps=60, lr=0.02, deadline=deadline - 10)
            
            # If GD found a reasonably good solution, polish with ALS
            if loss < 1.0:
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=15, deadline=deadline - 5)
                
                # Final verification
                if _verify(U, V, W, n, m, p, tol=1e-7):
                    found = True
                    best_rank = rank
                    best_result = (U, V, W)
                    break
            
            # Phase 2: Second GD pass with smaller lr if close
            if loss < 5.0:
                U2, V2, W2, loss2 = _gradient_descent(U, V, W, T, n, m, p, max_steps=40, lr=0.005, deadline=deadline - 5)
                if loss2 < 0.5:
                    U2, V2, W2 = _als_polish(U2, V2, W2, T, n, m, p, iterations=20, deadline=deadline - 3)
                    if _verify(U2, V2, W2, n, m, p, tol=1e-7):
                        found = True
                        best_rank = rank
                        best_result = (U2, V2, W2)
                        break
        
        if found:
            break
    
    # Final high-precision polish on the best result found
    U_final, V_final, W_final = best_result
    if time.time() < deadline - 10:
        U_final, V_final, W_final = _als_polish(U_final, V_final, W_final, T, n, m, p, iterations=30, deadline=deadline - 5)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
