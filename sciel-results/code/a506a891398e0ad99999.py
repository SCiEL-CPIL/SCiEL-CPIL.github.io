"""Tensor decomposition for 2x2 * 2x14 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Uses Strassen's algorithm applied to each of 7 non-overlapping 2x2 blocks of the
    second matrix. Total rank = 7 * 7 = 49.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7

    # Strassen's 7 bilinear terms for 2x2 * 2x2:
    # Each term is (a_coeffs, b_coeffs, c_coeffs) as sparse dicts.
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            # U[i*m + j, col] = coeff of A[i,j]
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            # V[j*p + k, col] = coeff of B[j,k] for block t
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            # W[k*n + i, col] = coeff of C[i,k] for block t
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify correctness
    def tensor_check(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s_val = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s_val - 1.0) > 1e-9:
                        return False
        return True

    if not tensor_check(U, V, W):
        # Fallback to schoolbook rank-56
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Convert to lists with integer entries where possible
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
