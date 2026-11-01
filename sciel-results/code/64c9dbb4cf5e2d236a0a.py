"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # We'll use rank 58: 2 Strassen blocks (7 each) + 44 standard terms = 58
    # Actually: 2*7 + (60 - 2*8) = 14 + 44 = 58
    max_rank = 58
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= max_rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Strassen block for 2x2x2 sub-block:
    # A rows i∈{0,1}, cols j∈{ja,jb}
    # B rows j∈{ja,jb}, cols k∈{ka,kb}
    # C rows k∈{ka,kb}, cols i∈{0,1}
    # 
    # U index for A[i,j]: i*m + j
    # V index for B[j,k]: j*p + k
    # W index for C[k,i]: k*n + i
    #
    # The 8 products we need to reproduce:
    # T[0,ja,ka] = A[0,ja]*B[ja,ka]*C[ka,0]
    # T[0,ja,kb] = A[0,ja]*B[ja,kb]*C[kb,0]
    # T[0,jb,ka] = A[0,jb]*B[jb,ka]*C[ka,0]
    # T[0,jb,kb] = A[0,jb]*B[jb,kb]*C[kb,0]
    # T[1,ja,ka] = A[1,ja]*B[ja,ka]*C[ka,1]
    # T[1,ja,kb] = A[1,ja]*B[ja,kb]*C[kb,1]
    # T[1,jb,ka] = A[1,jb]*B[jb,ka]*C[ka,1]
    # T[1,jb,kb] = A[1,jb]*B[jb,kb]*C[kb,1]
    #
    # Strassen: 7 multiplications
    # M1 = (A[0,ja]+A[1,jb]) * (B[ja,ka]+B[jb,kb])
    # M2 = (A[1,ja]+A[1,jb]) * B[ja,ka]
    # M3 = A[0,ja] * (B[ja,kb]-B[jb,kb])
    # M4 = A[1,jb] * (B[jb,ka]-B[ja,ka])
    # M5 = (A[0,ja]+A[0,jb]) * B[jb,kb]
    # M6 = (A[1,ja]-A[0,ja]) * (B[ja,ka]+B[ja,kb])
    # M7 = (A[0,jb]-A[1,jb]) * (B[jb,ka]+B[jb,kb])
    #
    # C entries:
    # C[ka,0] = M1 + M4 - M5 - M7
    # C[ka,1] = M2 + M1 - M4 - M6
    # C[kb,0] = M3 - M1 + M5 + M7
    # C[kb,1] = M2 + M6 - M3

    def strassen_block(ja, jb, ka, kb):
        # U indices
        u_0ja = 0 * m + ja   # A[0,ja]
        u_1ja = 1 * m + ja   # A[1,ja]
        u_0jb = 0 * m + jb   # A[0,jb]
        u_1jb = 1 * m + jb   # A[1,jb]
        
        # V indices
        v_jaka = ja * p + ka  # B[ja,ka]
        v_jakb = ja * p + kb  # B[ja,kb]
        v_jbka = jb * p + ka  # B[jb,ka]
        v_jbkb = jb * p + kb  # B[jb,kb]
        
        # W indices
        w_ka0 = ka * n + 0   # C[ka,0]
        w_ka1 = ka * n + 1   # C[ka,1]
        w_kb0 = kb * n + 0   # C[kb,0]
        w_kb1 = kb * n + 1   # C[kb,1]

        # M1 = (A[0,ja]+A[1,jb]) * (B[ja,ka]+B[jb,kb])
        add_term([(u_0ja, 1), (u_1jb, 1)], [(v_jaka, 1), (v_jbkb, 1)], [(w_ka0, 1), (w_kb0, -1)])
        
        # M2 = (A[1,ja]+A[1,jb]) * B[ja,ka]
        add_term([(u_1ja, 1), (u_1jb, 1)], [(v_jaka, 1)], [(w_ka1, 1)])
        
        # M3 = A[0,ja] * (B[ja,kb]-B[jb,kb])
        add_term([(u_0ja, 1)], [(v_jakb, 1), (v_jbkb, -1)], [(w_kb0, 1), (w_kb1, -1)])
        
        # M4 = A[1,jb] * (B[jb,ka]-B[ja,ka])
        add_term([(u_1jb, 1)], [(v_jbka, 1), (v_jaka, -1)], [(w_ka0, 1), (w_ka1, -1)])
        
        # M5 = (A[0,ja]+A[0,jb]) * B[jb,kb]
        add_term([(u_0ja, 1), (u_0jb, 1)], [(v_jbkb, 1)], [(w_ka0, -1), (w_kb0, 1)])
        
        # M6 = (A[1,ja]-A[0,ja]) * (B[ja,ka]+B[ja,kb])
        add_term([(u_1ja, 1), (u_0ja, -1)], [(v_jaka, 1), (v_jakb, 1)], [(w_ka1, -1), (w_kb1, 1)])
        
        # M7 = (A[0,jb]-A[1,jb]) * (B[jb,ka]+B[jb,kb])
        add_term([(u_0jb, 1), (u_1jb, -1)], [(v_jbka, 1), (v_jbkb, 1)], [(w_ka0, -1), (w_kb0, 1)])

    # Apply 2 Strassen blocks:
    # Block 1: j∈{0,1}, k∈{0,1}
    strassen_block(0, 1, 0, 1)
    # Block 2: j∈{2,3}, k∈{0,1}
    strassen_block(2, 3, 0, 1)

    # Track which (i,j,k) triples are covered by Strassen
    strassen_set = set()
    for i in range(n):
        for j in [0, 1, 2, 3]:
            for k in [0, 1]:
                strassen_set.add((i, j, k))

    # Standard terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_set:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verify correctness
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
        # Fallback to trivial rank 60
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
