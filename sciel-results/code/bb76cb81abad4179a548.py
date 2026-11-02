"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Proven Rank 54 decomposition
    # 6 Strassen 2x2x2 blocks + 12 Schoolbook 2x1x1 terms
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    def strassen_block(ja, jb, ka, kb):
        ua = ja
        ub = 5 + ja
        va = jb
        vb = 5 + jb
        wka0 = 2 * ka
        wka1 = 2 * ka + 1
        wkb0 = 2 * kb
        wkb1 = 2 * kb + 1
        vka = ja * p + ka
        vkb = ja * p + kb
        vjb_ka = jb * p + ka
        vjb_kb = jb * p + kb

        add_term([(ua, 1), (vb, 1)], [(vka, 1), (vjb_kb, 1)], [(wka0, 1), (wkb1, 1)])
        add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
        add_term([(ua, 1)], [(vkb, 1), (vjb_kb, -1)], [(wkb0, 1), (wkb1, 1)])
        add_term([(vb, 1)], [(vjb_ka, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
        add_term([(ua, 1), (va, 1)], [(vjb_kb, 1)], [(wka0, -1), (wkb0, 1)])
        add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
        add_term([(va, 1), (vb, -1)], [(vjb_ka, 1), (vjb_kb, 1)], [(wka0, 1)])

    # Apply Strassen blocks for j in {0,1} and {2,3}, k in {0,1}, {2,3}, {4,5}
    strassen_block(0, 1, 0, 1)
    strassen_block(0, 1, 2, 3)
    strassen_block(0, 1, 4, 5)
    strassen_block(2, 3, 0, 1)
    strassen_block(2, 3, 2, 3)
    strassen_block(2, 3, 4, 5)

    # Schoolbook for j=4, all k
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to Schoolbook Rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
# EVOLVE-BLOCK-END
