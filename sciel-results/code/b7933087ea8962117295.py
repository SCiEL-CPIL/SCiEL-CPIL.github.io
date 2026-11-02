import numpy as np
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
TIME_LIMIT = 110.0  # 150s limit, keep 40s margin

def _check_deadline():
    if time.time() - _START_TIME > TIME_LIMIT:
        raise TimeoutError("Deadline exceeded")

# The 7 Strassen linear combinations for A, B, and C.
_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 1, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition directly.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Outer Strassen loop
    for t_outer in range(7):
        # Inner Strassen loop
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # A (U) part
            for (br, bc, sign) in _A_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _A_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += sign * sign_inner
            
            # B (V) part
            for (br, bc, sign) in _B_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _B_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = i_global * n + k_global
                    V[v_idx, r_idx] += sign * sign_inner
            
            # C (W) part
            for (br, bc, sign) in _C_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _C_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += sign * sign_inner

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))

def construct():
    n, m, p = 4, 4, 4
    U, V, W = _generate_strassen_4x4()
    
    # Verify the rank-49 decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (should not happen)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
