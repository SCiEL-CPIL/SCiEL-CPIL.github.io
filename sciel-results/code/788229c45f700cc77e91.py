import numpy as np
# EVOLVE-BLOCK-START

# The 7 Strassen linear combinations for 2x2.
# Each entry is a list of (row, col, coeff) for the 2x2 block.
_A2 = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B2 = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C2 = [
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
    Generates U, V, W for 4x4 Strassen decomposition (rank 49).
    Uses two levels of Strassen: outer 2x2 blocks of size 2, inner 2x2 blocks of size 1.
    """
    n = 4
    rank = 49
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)

    # Outer Strassen: 2x2 blocks
    # Inner Strassen: 2x2 sub-blocks within each 2x2 block
    for t_out in range(7):
        for t_in in range(7):
            r_idx = t_out * 7 + t_in

            # U (A) part
            for (br, bc, s_out) in _A2[t_out]:
                for (ir, ik, s_in) in _A2[t_in]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += s_out * s_in

            # V (B) part
            for (br, bc, s_out) in _B2[t_out]:
                for (ir, ik, s_in) in _B2[t_in]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = i_global * n + k_global
                    V[v_idx, r_idx] += s_out * s_in

            # W (C) part
            for (br, bc, s_out) in _C2[t_out]:
                for (ir, ik, s_in) in _C2[t_in]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += s_out * s_in

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

    # Primary: rank-49 Strassen
    try:
        U, V, W = _generate_strassen_4x4()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank 64
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
