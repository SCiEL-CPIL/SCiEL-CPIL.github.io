"""Matrix multiplication 6x6x6: rank-189 decomposition via 2x2 Strassen outer × 3x3 schoolbook inner."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140

def _verify(U, V, W, n, m, p, tol=1e-10):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _schoolbook_6x6():
    n = m = p = 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_outer_schoolbook_inner():
    """
    Decompose 6x6 matrix multiplication as:
    - Outer: 2x2 block structure, each block is 3x3
    - Apply Strassen's 7-product algorithm at the block level
    - Each block product is computed via 3x3 schoolbook (27 scalar products)
    Total rank = 7 * 27 = 189
    """
    n = m = p = 6
    bs = 3  # block size
    nb = 2  # number of blocks per dimension

    rank = 7 * (bs ** 3)  # 189

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen's 7 products for 2x2 block matrix multiplication:
    # Let A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each "product" is actually a 3x3 matrix multiplication (schoolbook, 27 terms).

    # Define the 7 Strassen steps with their A-block, B-block, and C-block coefficients.
    # Each entry is a dict mapping (block_row, block_col) -> coefficient
    steps = [
        {
            'A': {(0, 0): 1, (1, 1): 1},
            'B': {(0, 0): 1, (1, 1): 1},
            'C': [(0, 0, 1), (1, 1, 1)]  # C11 += M1, C22 += M1
        },
        {
            'A': {(1, 0): 1, (1, 1): 1},
            'B': {(0, 0): 1},
            'C': [(1, 0, 1), (1, 1, -1)]  # C21 += M2, C22 -= M2
        },
        {
            'A': {(0, 0): 1},
            'B': {(0, 1): 1, (1, 1): -1},
            'C': [(0, 1, 1), (1, 1, 1)]  # C12 += M3, C22 += M3
        },
        {
            'A': {(1, 1): 1},
            'B': {(1, 0): 1, (0, 0): -1},
            'C': [(0, 0, 1), (1, 0, 1)]  # C11 += M4, C21 += M4
        },
        {
            'A': {(0, 0): 1, (0, 1): 1},
            'B': {(1, 1): 1},
            'C': [(0, 0, -1), (0, 1, 1)]  # C11 -= M5, C12 += M5
        },
        {
            'A': {(1, 0): 1, (0, 0): -1},
            'B': {(0, 0): 1, (0, 1): 1},
            'C': [(1, 1, 1)]  # C22 += M6
        },
        {
            'A': {(0, 1): 1, (1, 1): -1},
            'B': {(1, 0): 1, (1, 1): 1},
            'C': [(0, 0, 1)]  # C11 += M7
        },
    ]

    col = 0
    for step in steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_list = step['C']

        # For each of the 27 scalar products in the 3x3 schoolbook inner multiplication
        for u in range(bs):
            for v in range(bs):
                for w in range(bs):
                    # U: for each A-block (I,J) with coefficient ca, set U[(I*bs+u)*6 + (J*bs+v), col] += ca
                    for (I, J), ca in A_coeffs.items():
                        i_global = I * bs + u
                        j_global = J * bs + v
                        U[i_global * n + j_global, col] += ca

                    # V: for each B-block (J,K) with coefficient cb, set V[(J*bs+v)*6 + (K*bs+w), col] += cb
                    for (J, K), cb in B_coeffs.items():
                        j_global = J * bs + v
                        k_global = K * bs + w
                        V[j_global * p + k_global, col] += cb

                    # W: for each C-block (I,K) with coefficient cc, set W[(K*bs+w)*6 + (I*bs+u), col] += cc
                    # W is indexed by (k, i) where k is output col, i is output row
                    # C[i][k] gets contribution, W index is k*n + i
                    for (I, K, cc) in C_list:
                        i_global = I * bs + u
                        k_global = K * bs + w
                        W[k_global * n + i_global, col] += cc

                    col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    n, m, p = 6, 6, 6

    # Try the rank-189 Strassen-outer / schoolbook-inner decomposition
    try:
        res = _strassen_outer_schoolbook_inner()
        if _verify(res["U"], res["V"], res["W"], n, m, p):
            # Convert to list-of-lists as required by interface
            U_list = [[int(x) if x == int(x) else x for x in row] for row in np.array(res["U"], dtype=float)]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in np.array(res["V"], dtype=float)]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in np.array(res["W"], dtype=float)]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass

    # Fallback: schoolbook rank 216
    res = _schoolbook_6x6()
    U_list = [[int(x) for x in row] for row in res["U"]]
    V_list = [[int(x) for x in row] for row in res["V"]]
    W_list = [[int(x) for x in row] for row in res["W"]]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
