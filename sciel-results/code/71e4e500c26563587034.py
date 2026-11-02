"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    deadline = time.time() + 125
    
    # Build the list of (a, b, c) indices where T[a,b,c] = 1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    
    def tensor_value(U, V, W, r):
        """Compute the tensor from the decomposition."""
        T = np.zeros((nm, mp, pn))
        for (a, b, c) in indices:
            T[a, b, c] = np.sum(U[a] * V[b] * W[c])
        return T
    
    def compute_error(U, V, W, r):
        max_err = 0.0
        for (a, b, c) in indices:
            val = np.sum(U[a] * V[b] * W[c])
            max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def try_rank_als(r, seed, max_iter=5000):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            for a in range(nm):
                eqs = [(b, c) for (aa, b, c) in indices if aa == a]
                if not eqs:
                    continue
                A_mat = np.array([V[b] * W[c] for (b, c) in eqs])
                b_vec = np.ones(len(eqs))
                U[a], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V
            for b in range(mp):
                eqs = [(a, c) for (a, bb, c) in indices if bb == b]
                if not eqs:
                    continue
                A_mat = np.array([U[a] * W[c] for (a, c) in eqs])
                b_vec = np.ones(len(eqs))
                V[b], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W
            for c in range(pn):
                eqs = [(a, b) for (a, b, cc) in indices if cc == c]
                if not eqs:
                    continue
                A_mat = np.array([U[a] * V[b] for (a, b) in eqs])
                b_vec = np.ones(len(eqs))
                W[c], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            if it % 50 == 0:
                err = compute_error(U, V, W, r)
                if err < 1e-10:
                    # Try rounding
                    for scale in [2, 4, 8, 16]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        if compute_error(U_r, V_r, W_r, r) < 1e-9:
                            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if compute_error(U_i, V_i, W_i, r) < 1e-9:
                        return (U_i.tolist(), V_i.tolist(), W_i.tolist())
        
        return None
    
    # Try various ranks and seeds
    for r in range(12, 28):
        if time.time() > deadline:
            break
        for seed in range(60):
            if time.time() > deadline:
                break
            result = try_rank_als(r, seed)
            if result is not None:
                U_list, V_list, W_list = result
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
