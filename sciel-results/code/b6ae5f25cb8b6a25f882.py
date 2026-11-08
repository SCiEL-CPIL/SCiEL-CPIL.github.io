"""
5x5 matrix multiplication - correct rank decomposition.
Uses Strassen for 2x2 blocks (rank 7) and schoolbook for remaining blocks.
Block structure: split 5 as 2+3, then 3 as 2+1.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n * m, 1), dtype=float)
        V = np.zeros((m * p, 1), dtype=float)
        W = np.zeros((p * n, 1), dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j, 0] = val
        for (j, k), val in v_vals.items():
            V[j * p + k, 0] = val
        for (k, i), val in w_vals.items():
            W[k * n + i, 0] = val
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    def schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for A[i_list, j_list] * B[j_list, k_list] -> C[i_list, k_list]"""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def mul_3x3_rank23(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        """
        3x3 matrix multiplication with rank 23 (Bini-Capovani-Lotti-Romani 1979).
        A rows: i0,i1,i2; A cols=B rows: j0,j1,j2; B cols: k0,k1,k2; C rows: i0,i1,i2; C cols: k0,k1,k2
        """
        # M1 = (A00+A11)*(B00+B11)
        add_term({(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(k0,i0):1,(k1,i1):1})
        # M2 = (A10+A11)*B00
        add_term({(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(k0,i1):1,(k1,i1):-1})
        # M3 = A00*(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,i0):1,(k1,i1):1})
        # M4 = A11*(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,i0):1,(k0,i1):1})
        # M5 = (A00+A01)*B11
        add_term({(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1,(k1,i0):1})
        # M6 = (A10-A00)*(B00+B01)
        add_term({(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)*(B10+B11)
        add_term({(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(k0,i0):1})
        # M8 = A22*(B00+B11+B22)
        add_term({(i2,j2):1}, {(j0,k0):1,(j1,k1):1,(j2,k2):1}, {(k0,i2):1,(k1,i2):1,(k2,i2):1})
        # M9 = (A02+A12)*(B00+B11)
        add_term({(i0,j2):1,(i1,j2):1}, {(j0,k0):1,(j1,k1):1}, {(k0,i0):1,(k1,i1):1})
        # M10 = (A02+A12)*(B01-B11)
        add_term({(i0,j2):1,(i1,j2):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,i0):1,(k1,i1):1})
        # M11 = (A02+A12)*(B10-B00)
        add_term({(i0,j2):1,(i1,j2):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,i0):1,(k0,i1):1})
        # M12 = (A20+A21)*B22
        add_term({(i2,j0):1,(i2,j1):1}, {(j2,k2):1}, {(k2,i2):1})
        # M13 = (A20+A21)*(B02-B12)
        add_term({(i2,j0):1,(i2,j1):1}, {(j0,k2):1,(j1,k2):-1}, {(k2,i2):1})
        # M14 = (A20+A21)*(B12-B02)
        add_term({(i2,j0):1,(i2,j1):1}, {(j1,k2):1,(j0,k2):-1}, {(k2,i2):1})
        # M15 = (A00+A22)*(B11+B22)
        add_term({(i0,j0):1,(i2,j2):1}, {(j1,k1):1,(j2,k2):1}, {(k1,i0):1,(k2,i2):1})
        # M16 = (A11+A22)*(B00+B22)
        add_term({(i1,j1):1,(i2,j2):1}, {(j0,k0):1,(j2,k2):1}, {(k0,i1):1,(k2,i2):1})
        # M17 = (A01+A22)*(B10+B22)
        add_term({(i0,j1):1,(i2,j2):1}, {(j1,k0):1,(j2,k2):1}, {(k0,i0):1,(k2,i2):1})
        # M18 = (A00+A11+A22)*(B00+B11+B22) - (M1 + M8 contributions) -- this gets complex
        # Let me use a different, verified rank-23 formulation
        pass

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """
        Strassen decomposition for A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]
        where i0,i1 are the two row indices, j0,j1 the two col indices of A (and row indices of B),
        k0,k1 the two output column indices.
        """
        # M1 = (A00+A11)*(B00+B11)
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})
        # M2 = (A10+A11)*B00
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})
        # M3 = A00*(B01-B11)
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})
        # M4 = A11*(B10-B00)
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})
        # M5 = (A00+A01)*B11
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})
        # M6 = (A10-A00)*(B00+B01)
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})
        # M7 = (A01-A11)*(B10+B11)
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    # Block structure: 5 = 2 + 2 + 1
    # Block sizes: s0=2 (indices 0,1), s1=2 (indices 2,3), s2=1 (index 4)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.

    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2] + A[0:2,4]*B[4,0:2]
    strassen_2x2(0, 1, 0, 1, 0, 1)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    schoolbook_block([0, 1], [4], [0, 1])

    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4] + A[0:2,4]*B[4,2:4]
    strassen_2x2(0, 1, 0, 1, 2, 3)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    schoolbook_block([0, 1], [4], [2, 3])

    # C[0:2, 4] = A[0:2,0:2]*B[0:2,4] + A[0:2,2:4]*B[2:4,4] + A[0:2,4]*B[4,4]
    schoolbook_block([0, 1], [0, 1], [4])
    schoolbook_block([0, 1], [2, 3], [4])
    schoolbook_block([0, 1], [4], [4])

    # C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2] + A[2:4,4]*B[4,0:2]
    strassen_2x2(2, 3, 0, 1, 0, 1)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    schoolbook_block([2, 3], [4], [0, 1])

    # C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4] + A[2:4,4]*B[4,2:4]
    strassen_2x2(2, 3, 0, 1, 2, 3)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    schoolbook_block([2, 3], [4], [2, 3])

    # C[2:4, 4] = A[2:4,0:2]*B[0:2,4] + A[2:4,2:4]*B[2:4,4] + A[2:4,4]*B[4,4]
    schoolbook_block([2, 3], [0, 1], [4])
    schoolbook_block([2, 3], [2, 3], [4])
    schoolbook_block([2, 3], [4], [4])

    # C[4, 0:2] = A[4,0:2]*B[0:2,0:2] + A[4,2:4]*B[2:4,0:2] + A[4,4]*B[4,0:2]
    schoolbook_block([4], [0, 1], [0, 1])
    schoolbook_block([4], [2, 3], [0, 1])
    schoolbook_block([4], [4], [0, 1])

    # C[4, 2:4] = A[4,0:2]*B[0:2,2:4] + A[4,2:4]*B[2:4,2:4] + A[4,4]*B[4,2:4]
    schoolbook_block([4], [0, 1], [2, 3])
    schoolbook_block([4], [2, 3], [2, 3])
    schoolbook_block([4], [4], [2, 3])

    # C[4, 4] = A[4,0:2]*B[0:2,4] + A[4,2:4]*B[2:4,4] + A[4,4]*B[4,4]
    schoolbook_block([4], [0, 1], [4])
    schoolbook_block([4], [2, 3], [4])
    schoolbook_block([4], [4], [4])

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
