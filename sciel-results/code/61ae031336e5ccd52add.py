"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                target_indices.append((a, b, c))
    
    def verify_decomp(U, V, W):
        """Verify that U,V,W correctly decompose the multiplication tensor."""
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_comp, T_target, atol=1e-8)
    
    def try_rank_ls(r, seed=0, n_attempts=5):
        """Try to find rank-r decomposition by fixing U,V and solving for W via least squares."""
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed + attempt * 1000)
            U = rng.randn(nm, r)
            V = rng.randn(mp, r)
            
            W = np.zeros((pn, r))
            for c in range(pn):
                # Build linear system: for all (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                A_c = np.zeros((nm * mp, r))
                t_c = np.zeros(nm * mp)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        A_c[idx, :] = U[a, :] * V[b, :]
                        t_c[idx] = T_target[a, b, c]
                        idx += 1
                w_c, _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
                W[c, :] = w_c
            
            if verify_decomp(U, V, W):
                return U, V, W
        return None
    
    def try_rank_structured(r, seed=0):
        """Try structured initialization for rank-r decomposition."""
        rng = np.random.RandomState(seed)
        # Initialize U, V with sparse structure
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        for col in range(r):
            # Each column of U has a few nonzeros
            nnz_u = max(1, min(nm, rng.randint(1, 4)))
            rows_u = rng.choice(nm, nnz_u, replace=False)
            for row in rows_u:
                U[row, col] = rng.choice([-1, 1])
            nnz_v = max(1, min(mp, rng.randint(1, 4)))
            rows_v = rng.choice(mp, nnz_v, replace=False)
            for row in rows_v:
                V[row, col] = rng.choice([-1, 1])
        
        W = np.zeros((pn, r))
        for c in range(pn):
            A_c = np.zeros((nm * mp, r))
            t_c = np.zeros(nm * mp)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    A_c[idx, :] = U[a, :] * V[b, :]
                    t_c[idx] = T_target[a, b, c]
                    idx += 1
            w_c, _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
            W[c, :] = w_c
        
        if verify_decomp(U, V, W):
            return U, V, W
        return None
    
    # Try various ranks from low to high
    best_result = None
    for r in range(40, 55):
        if time.time() > deadline:
            break
        # Try structured first
        for seed in range(3):
            if time.time() > deadline:
                break
            result = try_rank_structured(r, seed=seed)
            if result is not None:
                best_result = (r, result)
                break
        if best_result is not None and best_result[0] == r:
            break
        # Try random
        for seed in range(3):
            if time.time() > deadline:
                break
            result = try_rank_ls(r, seed=seed, n_attempts=2)
            if result is not None:
                best_result = (r, result)
                break
        if best_result is not None and best_result[0] == r:
            break
    
    if best_result is not None:
        r, (U, V, W) = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook (rank = n*m*p = 54)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb), dtype=np.float64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.float64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
