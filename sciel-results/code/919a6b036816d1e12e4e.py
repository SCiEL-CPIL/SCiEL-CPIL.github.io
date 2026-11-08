"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B."""
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i*m+j, j*p+k, k*n+i] != 1:
                    return False
    required = set((i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if T[a,b,c] != 0 and (a,b,c) not in required:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decomp(T, n, m, p, rank, seed=42, max_iters=200, tol=1e-8, deadline=None):
    """Vectorized ALS for rank-r decomposition of tensor T."""
    np.random.seed(seed)
    A_dim, B_dim, C_dim = n*m, m*p, p*n
    
    U = np.random.randn(A_dim, rank)
    V = np.random.randn(B_dim, rank)
    W = np.random.randn(C_dim, rank)
    
    T_a = T.reshape(A_dim, -1)
    T_b = T.reshape(B_dim, -1)
    T_c = T.reshape(C_dim, -1)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each r, solve U[:,r] = argmin ||T - sum_r' U[:,r']*V[:,r']*W[:,r']||
        # M_U[r, (b,c)] = V[b,r]*W[c,r]
        M_U = (V[:, :, None] * W[:, None, :]).reshape(rank, B_dim * C_dim)
        U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_a.T, rcond=None)
        U = U_new.T
        
        # Update V
        M_V = (U[:, :, None] * W[:, None, :]).reshape(rank, A_dim * C_dim)
        V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_b.T, rcond=None)
        V = V_new.T
        
        # Update W
        M_W = (U[:, :, None] * V[:, None, :]).reshape(rank, A_dim * B_dim)
        W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_c.T, rcond=None)
        W = W_new.T
        
        # Check convergence
        if it % 5 == 0 or it == max_iters - 1:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err > 1e-6:
        return None
    return U, V, W

def _rank55(n, m, p, deadline=None):
    """Block-based rank-55 decomposition for 2x3 x 3x10."""
    if n != 2 or m != 3 or p != 10:
        return None
    
    # Find rank-11 decomposition for 2x3 x 3x2 sub-problem
    T_sub = _build_target_tensor(2, 3, 2)
    
    sub = None
    seeds = [42, 1, 7, 13, 99, 123, 77, 200, 314, 500]
    for seed in seeds:
        if deadline is not None and time.time() > deadline - 10:
            break
        sub = _als_decomp(T_sub, 2, 3, 2, 11, seed=seed, max_iters=200, deadline=deadline)
        if sub is not None:
            break
    
    if sub is None:
        return None
    
    U_sub, V_sub, W_sub = sub
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t in range(5):
        k0 = 2*t
        for r in range(11):
            col = t*11 + r
            U[:, col] += U_sub[:, r]
            for j in range(3):
                for k in range(2):
                    b_sub = j*2 + k
                    b_full = j*10 + k0 + k
                    V[b_full, col] += V_sub[b_sub, r]
            for k in range(2):
                for i in range(2):
                    c_sub = k*2 + i
                    c_full = (k0 + k)*2 + i
                    W[c_full, col] += W_sub[c_sub, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p, deadline=deadline)
        if res55 is not None and time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
