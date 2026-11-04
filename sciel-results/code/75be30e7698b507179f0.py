"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Fast verification: check all non-zero tensor entries + sample zeros."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
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
    for _ in range(300):
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

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification checking all zero entries."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
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
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
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

def _bini_capovani_newton(U, V, W, T, n, m, p, max_iter=50, tol=1e-12, deadline=None):
    """
    Bini-Capovani Newton-Raphson iteration for tensor decomposition.
    At each step, fix one tensor and solve linear least squares for the others.
    This is essentially ALS but with careful convergence criteria.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute residuals
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < tol:
            break
        
        # Update U: for each a, solve U[a,:] @ (V*W)[bc,:]^T = T[a,b,c]
        # (V*W)[bc,r] = V[b,r]*W[c,r]
        VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, r)
        VW_flat = VW.reshape(mp * pn, r)  # (mp*pn, r)
        
        for a in range(nm):
            T_a = T[a, :, :].flatten()  # (mp*pn,)
            # U[a,:] = (VW_flat^T @ VW_flat)^{-1} @ VW_flat^T @ T_a
            A = VW_flat.T  # (r, mp*pn)
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(A, T_a, rcond=None)
            except:
                pass
        
        # Update V: for each b, solve V[b,:] @ (U*W)[ac,:]^T = T[a,b,c]
        UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, r)
        UW_flat = UW.reshape(nm * pn, r)
        
        for b in range(mp):
            T_b = T[:, b, :].flatten()  # (nm*pn,)
            A = UW_flat.T  # (r, nm*pn)
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(A, T_b, rcond=None)
            except:
                pass
        
        # Update W: for each c, solve W[c,:] @ (U*V)[ab,:]^T = T[a,b,c]
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, r)
        UV_flat = UV.reshape(nm * mp, r)
        
        for c in range(pn):
            T_c = T[:, :, c].flatten()  # (nm*mp,)
            A = UV_flat.T  # (r, nm*mp)
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(A, T_c, rcond=None)
            except:
                pass
    
    return U, V, W

def _structured_init(n, m, p, rank, seed=42, strategy='random'):
    """Create structured initialization exploiting the tensor's structure."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    if strategy == 'block':
        # Block structure: group by j index
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        if rank >= m:
            terms_per_j = rank // m
            remainder = rank % m
            col = 0
            for j in range(m):
                n_terms = terms_per_j + (1 if j < remainder else 0)
                for t in range(n_terms):
                    for i in range(n):
                        U[i*m+j, col] = rng.randn()
                    for k in range(p):
                        V[j*p+k, col] = rng.randn()
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, col] = rng.randn()
                    col += 1
        return U, V, W
    
    elif strategy == 'sparse':
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Each factor has a fixed sparsity pattern
        for a in range(nm):
            nnz = max(1, int(rank * 0.4))
            cols = rng.choice(rank, nnz, replace=False)
            U[a, cols] = rng.randn(nnz)
        
        for b in range(mp):
            nnz = max(1, int(rank * 0.4))
            cols = rng.choice(rank, nnz, replace=False)
            V[b, cols] = rng.randn(nnz)
        
        for c in range(pn):
            nnz = max(1, int(rank * 0.4))
            cols = rng.choice(rank, nnz, replace=False)
            W[c, cols] = rng.randn(nnz)
        
        return U, V, W
    
    elif strategy == 'very_sparse':
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for a in range(nm):
            nnz = max(1, int(rank * 0.2))
            cols = rng.choice(rank, nnz, replace=False)
            U[a, cols] = rng.randn(nnz)
        
        for b in range(mp):
            nnz = max(1, int(rank * 0.2))
            cols = rng.choice(rank, nnz, replace=False)
            V[b, cols] = rng.randn(nnz)
        
        for c in range(pn):
            nnz = max(1, int(rank * 0.2))
            cols = rng.choice(rank, nnz, replace=False)
            W[c, cols] = rng.randn(nnz)
        
        return U, V, W
    
    else:  # random
        scale = 1.0 / np.sqrt(rank)
        U = rng.randn(nm, rank) * scale
        V = rng.randn(mp, rank) * scale
        W = rng.randn(pn, rank) * scale
        return U, V, W

def _gradient_descent(U, V, W, T, n, m, p, max_steps=80, lr=0.02, deadline=None):
    """Gradient descent with adaptive learning rate."""
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
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook fallback: rank = n*m*p = 60
    best_rank = n * m * p
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
    
    # Lower bound: max(nm, mp, pn) = max(10, 30, 12) = 30
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_restarts = 8
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Choose initialization strategy based on restart
            if restart % 4 == 0:
                strategy = 'block'
            elif restart % 4 == 1:
                strategy = 'random'
            elif restart % 4 == 2:
                strategy = 'sparse'
            else:
                strategy = 'very_sparse'
            
            seed = 100 + restart * 13 + rank * 7
            U, V, W = _structured_init(n, m, p, rank, seed=seed, strategy=strategy)
            
            # Phase 1: Gradient descent to get into basin of attraction
            U, V, W, loss = _gradient_descent(U, V, W, T, n, m, p, max_steps=50, lr=0.02, deadline=deadline - 15)
            
            # Phase 2: Bini-Capovani Newton iterations (ALS)
            if loss < 2.0:
                U, V, W = _bini_capovani_newton(U, V, W, T, n, m, p, max_iter=30, tol=1e-12, deadline=deadline - 8)
                
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-7):
                    # Full verification for safety
                    if _verify_full(U, V, W, n, m, p, tol=1e-8):
                        found = True
                        best_rank = rank
                        best_result = (U, V, W)
                        break
            
            # If not found, try a second GD pass with smaller lr
            if loss < 5.0 and not found:
                U2, V2, W2, loss2 = _gradient_descent(U, V, W, T, n, m, p, max_steps=30, lr=0.005, deadline=deadline - 8)
                if loss2 < 1.0:
                    U2, V2, W2 = _bini_capovani_newton(U2, V2, W2, T, n, m, p, max_iter=20, tol=1e-12, deadline=deadline - 5)
                    if _verify(U2, V2, W2, n, m, p, tol=1e-7):
                        if _verify_full(U2, V2, W2, n, m, p, tol=1e-8):
                            found = True
                            best_rank = rank
                            best_result = (U2, V2, W2)
                            break
        
        if found:
            break
    
    # Final polish on best result
    U_final, V_final, W_final = best_result
    if time.time() < deadline - 5:
        U_final, V_final, W_final = _bini_capovani_newton(U_final, V_final, W_final, T, n, m, p, max_iter=20, tol=1e-14, deadline=deadline - 2)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
