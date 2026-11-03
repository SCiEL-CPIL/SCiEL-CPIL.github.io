import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=100, tol=1e-10, seed=0, time_budget=None, U_init=None, V_init=None, W_init=None):
    """
    Alternating Least Squares to decompose T into U, V, W.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    a, b, c = T.shape
    
    if U_init is not None:
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
    else:
        scale = 1.0 / np.sqrt(rank)
        U = rng.randn(a, rank) * scale
        V = rng.randn(b, rank) * scale
        W = rng.randn(c, rank) * scale
        
    # Precompute flattened target views
    T_ac = T.reshape(a, b * c)
    T_bc = T.transpose(1, 0, 2).reshape(b, a * c)
    T_cc = T.transpose(2, 0, 1).reshape(c, a * b)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
            
        # Update U
        Khatri_VW = np.einsum('br,cr->bcr', V, W).reshape(b * c, rank).T
        try:
            U_new = np.linalg.lstsq(Khatri_VW, T_ac.T, rcond=None)[0].T
        except:
            U_new = U
        
        # Update V
        Khatri_UW = np.einsum('ar,cr->acr', U_new, W).reshape(a * c, rank).T
        try:
            V_new = np.linalg.lstsq(Khatri_UW, T_bc.T, rcond=None)[0].T
        except:
            V_new = V
        
        # Update W
        Khatri_UV = np.einsum('ar,br->abr', U_new, V_new).reshape(a * b, rank).T
        try:
            W_new = np.linalg.lstsq(Khatri_UV, T_cc.T, rcond=None)[0].T
        except:
            W_new = W
        
        U_diff = np.linalg.norm(U_new - U)
        V_diff = np.linalg.norm(V_new - V)
        W_diff = np.linalg.norm(W_new - W)
        
        U, V, W = U_new, V_new, W_new
        
        if U_diff + V_diff + W_diff < tol:
            break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    return U, V, W, residual

def _find_decomp_small(n, m, p, rank, max_restarts=20, seed_base=0):
    """Find a low-rank decomposition for a small tensor."""
    T = _build_target_tensor(n, m, p)
    best_U, best_V, best_W, best_res = None, None, None, float('inf')
    
    for r in range(max_restarts):
        seed = seed_base + r
        try:
            U, V, W, res = _als_decompose(T, rank, max_iter=50, tol=1e-12, seed=seed)
            if res < best_res:
                best_U, best_V, best_W, best_res = U, V, W, res
            if res < 1e-8:
                break
        except:
            continue
            
    if best_res < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 135.0
    
    # Strategy 1: Use structure of M_{2,3,9} as 3 blocks of M_{2,3,3}
    # Find rank-7 decomposition for M_{2,3,3}
    n0, m0, p0 = 2, 3, 3
    rank0 = 7
    
    U0, V0, W0 = _find_decomp_small(n0, m0, p0, rank0, max_restarts=10, seed_base=100)
    
    if U0 is not None:
        # Construct rank-21 decomposition for M_{2,3,9}
        rank_total = rank0 * 3
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        for block in range(3):
            offset = block * rank0
            # U is the same for all blocks?
            # A is 2x3. In M_{2,3,3}, U is 6x7.
            # In M_{2,3,9}, A is the same 2x3 matrix.
            # So U should be U0 repeated?
            # T = sum_r U_r V_r W_r
            # For block b, the terms are U0[r] V0_b[r] W0_b[r]
            # V0_b is V0 placed in the block b position.
            # W0_b is W0 placed in the block b position.
            # U is the same U0 for all blocks.
            U[:, offset:offset+rank0] = U0
            
            # V indices: j*p + k. j in 0..2, k in 0..8.
            # Block b corresponds to k in 3*b .. 3*b+2.
            # So V[j, 3*b + k'] = V0[j, k']
            for j in range(m0):
                for k_prime in range(p0):
                    V[j * p + (3 * block + k_prime), offset:offset+rank0] = V0[j * p0 + k_prime, :]
            
            # W indices: k*n + i. k in 0..8, i in 0..1.
            # Block b corresponds to k in 3*b .. 3*b+2.
            # So W[3*b + k', i] = W0[k', i]
            for k_prime in range(p0):
                for i in range(n0):
                    W[(3 * block + k_prime) * n + i, offset:offset+rank0] = W0[k_prime * n0 + i, :]
        
        # Verify
        T = _build_target_tensor(n, m, p)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
        
        if residual < 1e-8:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Strategy 2: Direct search for M_{2,3,9} with low ranks
    T = _build_target_tensor(n, m, p)
    
    for rank in range(21, 30):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            seed = rank * 10 + attempt
            try:
                U, V, W, res = _als_decompose(T, rank, max_iter=100, tol=1e-12, seed=seed, time_budget=deadline - 5)
                if res < 1e-8:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
            except:
                pass
    
    # Fallback: Schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
