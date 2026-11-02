"""Direct rank-minimization search for 2x2x11 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 11
    deadline = time.time() + 115.0
    
    # Build index arrays for the tensor
    ta = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    tb = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    tc = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def _verify(U, V, W, tol=1e-12):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _als_search(rank, n, m, p, ta, tb, tc, max_iters=2000, deadline=None, seed=42):
        """Accelerated alternating least squares for tensor rank minimization."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Precompute masks for efficient updates
        ta_groups = {}
        tb_groups = {}
        tc_groups = {}
        for idx in range(len(ta)):
            ta_groups.setdefault(ta[idx], []).append(idx)
            tb_groups.setdefault(tb[idx], []).append(idx)
            tc_groups.setdefault(tc[idx], []).append(idx)
        ta_groups = {k: np.array(v) for k, v in ta_groups.items()}
        tb_groups = {k: np.array(v) for k, v in tb_groups.items()}
        tc_groups = {k: np.array(v) for k, v in tc_groups.items()}
        
        def _error(U, V, W):
            s = np.zeros(len(ta), dtype=np.float64)
            for c in range(rank):
                s += U[ta, c] * V[tb, c] * W[tc, c]
            return float(np.sum((s - 1.0) ** 2))
        
        err = _error(U, V, W)
        best_err = err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                idxs = ta_groups[a]
                tb_sub = tb[idxs]
                tc_sub = tc[idxs]
                prod = V[tb_sub, :] * W[tc_sub, :]
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-10
                if np.any(valid):
                    U[a, valid] = num[valid] / den[valid]
            
            # Update V
            for b in range(m * p):
                idxs = tb_groups[b]
                ta_sub = ta[idxs]
                tc_sub = tc[idxs]
                prod = U[ta_sub, :] * W[tc_sub, :]
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-10
                if np.any(valid):
                    V[b, valid] = num[valid] / den[valid]
            
            # Update W
            for c_idx in range(p * n):
                idxs = tc_groups[c_idx]
                ta_sub = ta[idxs]
                tb_sub = tb[idxs]
                prod = U[ta_sub, :] * V[tb_sub, :]
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-10
                if np.any(valid):
                    W[c_idx, valid] = num[valid] / den[valid]
            
            if it % 50 == 0:
                err = _error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-14:
                    break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_err
    
    # Try to find a low-rank decomposition
    for rank in range(28, 40):
        if time.time() > deadline - 5:
            break
        for seed in [1, 7, 13, 42, 99, 123, 256, 512, 777, 1000, 2024, 3141, 4242, 5555, 6789, 8888, 9999, 12345]:
            if time.time() > deadline - 5:
                break
            U_l, V_l, W_l, err = _als_search(
                rank, n, m, p, ta, tb, tc,
                max_iters=3000, deadline=deadline - 10, seed=seed
            )
            if err < 1e-10:
                if _verify(U_l, V_l, W_l):
                    return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fallback: rank-44 schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
