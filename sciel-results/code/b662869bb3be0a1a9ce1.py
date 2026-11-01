import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Schoolbook rank 64 as fallback
    rank_school = n * m * p  # 64
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Strassen-based rank-56 decomposition
    # Partition A (2x4) into two 2x2 blocks along j
    # Partition B (4x8) into 8 2x2 blocks (2 along j, 4 along k)
    # Each 2x2 * 2x2 -> 2x2 uses Strassen rank 7
    # Total: 2 * 4 * 7 = 56

    rank_56 = 56
    U_56 = np.zeros((nm, rank_56))
    V_56 = np.zeros((mp, rank_56))
    W_56 = np.zeros((pn, rank_56))

    # Correct Strassen 7-term decomposition for 2x2x2 tensor
    # C = A * B, where A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (local_i, local_j, coeff) for 2x2 A block
    # B_coeffs: list of (local_i, local_j, coeff) for 2x2 B block
    # C_coeffs: list of (local_i, local_j, coeff) for 2x2 C block

    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (a10+a11)(b00)
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = (a00)(b01-b11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = (a11)(b10-b00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (a00+a01)(b11)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (a10-a00)(b00+b01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (a01-a11)(b10+b11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    col = 0
    for a_block in range(2):   # A-block index: j in {2*a_block, 2*a_block+1}
        for b_block in range(4):  # B-block index: k in {2*b_block, 2*b_block+1}
            for A_coeffs, B_coeffs, C_coeffs in strassen_terms:
                # Fill U: U[i*m + j, col]
                for (li, lj, coeff) in A_coeffs:
                    global_i = li
                    global_j = 2 * a_block + lj
                    U_56[global_i * m + global_j, col] += coeff

                # Fill V: V[j*p + k, col]
                for (lj, lk, coeff) in B_coeffs:
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V_56[global_j * p + global_k, col] += coeff

                # Fill W: W[k*n + i, col]
                for (li, lk, coeff) in C_coeffs:
                    global_i = li
                    global_k = 2 * b_block + lk
                    W_56[global_k * n + global_i, col] += coeff

                col += 1

    # Verify the rank-56 decomposition
    if time.time() < deadline:
        U_t = U_56.T.reshape(n, m, rank_56)
        V_t = V_56.T.reshape(m, p, rank_56)
        W_t = W_56.T.reshape(p, n, rank_56)
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        target = np.ones((n, m, p))
        if np.allclose(result, target, atol=1e-10):
            return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
