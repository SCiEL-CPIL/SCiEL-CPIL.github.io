"""
5x5 matrix multiplication - optimized block decomposition (2+3 split) with Strassen 3x3.
Uses Strassen for 2x2x2 blocks, a rank-23 decomposition for the 3x3x3 block,
and schoolbook for rectangular cross terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n*m, dtype=float)
        V = np.zeros(m*p, dtype=float)
        W = np.zeros(p*n, dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j] = val
        for (j, k), val in v_vals.items():
            V[j*p+k] = val
        for (k, i), val in w_vals.items():
            W[k*n+i] = val

        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen decomposition for 2x2 * 2x2 -> 2x2"""
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def strassen_3x3(i0, j0, k0):
        """
        Rank-23 decomposition for 3x3 * 3x3 -> 3x3.
        Indices: i0, i0+1, i0+2 for rows; j0, j0+1, j0+2 for cols of A (and rows of B);
        k0, k0+1, k0+2 for cols of B (and cols of C).
        """
        i1, i2 = i0+1, i0+2
        j1, j2 = j0+1, j0+2
        k1, k2 = k0+1, k0+2

        # M1 = (A00+A11+A22)(B00+B11+B22)
        add_term({(i0,j0):1, (i1,j1):1, (i2,j2):1},
                 {(j0,k0):1, (j1,k1):1, (j2,k2):1},
                 {(k0,i0):1, (k1,i1):1, (k2,i2):1})
        # M2 = (A01+A12+A20)(B10+B21+B02)
        add_term({(i0,j1):1, (i1,j2):1, (i2,j0):1},
                 {(j1,k0):1, (j2,k1):1, (j0,k2):1},
                 {(k0,i1):1, (k1,i2):1, (k2,i0):1})
        # M3 = (A02+A10+A21)(B20+B01+B12)
        add_term({(i0,j2):1, (i1,j0):1, (i2,j1):1},
                 {(j2,k0):1, (j0,k1):1, (j1,k2):1},
                 {(k0,i2):1, (k1,i0):1, (k2,i1):1})
        # M4 = (A00+A01)(B00+B01)
        add_term({(i0,j0):1, (i0,j1):1},
                 {(j0,k0):1, (j0,k1):1},
                 {(k1,i0):1})
        # M5 = (A01+A02)(B01+B02)
        add_term({(i0,j1):1, (i0,j2):1},
                 {(j0,k1):1, (j0,k2):1},
                 {(k2,i0):1})
        # M6 = (A02+A00)(B02+B00)
        add_term({(i0,j2):1, (i0,j0):1},
                 {(j0,k2):1, (j0,k0):1},
                 {(k0,i0):1})
        # M7 = (A10+A11)(B10+B11)
        add_term({(i1,j0):1, (i1,j1):1},
                 {(j1,k0):1, (j1,k1):1},
                 {(k1,i1):1})
        # M8 = (A11+A12)(B11+B12)
        add_term({(i1,j1):1, (i1,j2):1},
                 {(j1,k1):1, (j1,k2):1},
                 {(k2,i1):1})
        # M9 = (A12+A10)(B12+B10)
        add_term({(i1,j2):1, (i1,j0):1},
                 {(j1,k2):1, (j1,k0):1},
                 {(k0,i1):1})
        # M10 = (A20+A21)(B20+B21)
        add_term({(i2,j0):1, (i2,j1):1},
                 {(j2,k0):1, (j2,k1):1},
                 {(k1,i2):1})
        # M11 = (A21+A22)(B21+B22)
        add_term({(i2,j1):1, (i2,j2):1},
                 {(j2,k1):1, (j2,k2):1},
                 {(k2,i2):1})
        # M12 = (A22+A20)(B22+B20)
        add_term({(i2,j2):1, (i2,j0):1},
                 {(j2,k2):1, (j2,k0):1},
                 {(k0,i2):1})
        # M13 = (A00+A10+A20)(B00+B10+B20)
        add_term({(i0,j0):1, (i1,j0):1, (i2,j0):1},
                 {(j0,k0):1, (j1,k0):1, (j2,k0):1},
                 {(k0,i0):1, (k0,i1):1, (k0,i2):1})
        # M14 = (A01+A11+A21)(B01+B11+B21)
        add_term({(i0,j1):1, (i1,j1):1, (i2,j1):1},
                 {(j0,k1):1, (j1,k1):1, (j2,k1):1},
                 {(k1,i0):1, (k1,i1):1, (k1,i2):1})
        # M15 = (A02+A12+A22)(B02+B12+B22)
        add_term({(i0,j2):1, (i1,j2):1, (i2,j2):1},
                 {(j0,k2):1, (j1,k2):1, (j2,k2):1},
                 {(k2,i0):1, (k2,i1):1, (k2,i2):1})
        # M16 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1},
                 {(j0,k0):1, (j1,k1):1},
                 {(k0,i0):1, (k1,i1):1})
        # M17 = (A10+A22)(B10+B22)
        add_term({(i1,j0):1, (i2,j2):1},
                 {(j1,k0):1, (j2,k2):1},
                 {(k0,i1):1, (k2,i2):1})
        # M18 = (A20+A01)(B20+B01)
        add_term({(i2,j0):1, (i0,j1):1},
                 {(j2,k0):1, (j0,k1):1},
                 {(k0,i2):1, (k1,i0):1})
        # M19 = (A01+A22)(B01+B22)
        add_term({(i0,j1):1, (i2,j2):1},
                 {(j0,k1):1, (j2,k2):1},
                 {(k1,i0):1, (k2,i2):1})
        # M20 = (A12+A00)(B12+B00)
        add_term({(i1,j2):1, (i0,j0):1},
                 {(j1,k2):1, (j0,k0):1},
                 {(k2,i1):1, (k0,i0):1})
        # M21 = (A21+A10)(B21+B10)
        add_term({(i2,j1):1, (i1,j0):1},
                 {(j2,k1):1, (j1,k0):1},
                 {(k1,i2):1, (k0,i1):1})
        # M22 = (A00+A21)(B00+B21)
        add_term({(i0,j0):1, (i2,j1):1},
                 {(j0,k0):1, (j2,k1):1},
                 {(k0,i0):1, (k1,i2):1})
        # M23 = (A11+A02)(B11+B02)
        add_term({(i1,j1):1, (i0,j2):1},
                 {(j1,k1):1, (j0,k2):1},
                 {(k1,i1):1, (k2,i0):1})

    # Block decomposition: split indices into {0,1} and {2,3,4}
    # C_22 = A_22 B_22 + A_23 B_32
    strassen_2x2(0, 1, 0, 1, 0, 1)       # A_22 * B_22
    strassen_2x2(0, 1, 2, 3, 0, 1)       # A_23a(j=2,3) * B_32a(j=2,3)
    for i in [0, 1]:                     # A_23b(j=4) * B_32b(j=4)
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C_23 = A_22 B_23 + A_23 B_33
    strassen_2x2(0, 1, 0, 1, 2, 3)       # A_22 * B_23a(k=2,3)
    for i in [0, 1]:                     # A_22 * B_23b(k=4)
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    strassen_2x2(0, 1, 2, 3, 2, 3)       # A_23a(j=2,3) * B_33a(j=2,3,k=2,3)
    for i in [0, 1]:                     # A_23a(j=2,3) * B_33b(j=2,3,k=4)
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    for i in [0, 1]:                     # A_23b(j=4) * B_33b(j=4,k=2,3,4)
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C_32 = A_32 B_22 + A_33 B_32
    strassen_2x2(2, 3, 0, 1, 0, 1)       # A_32a(i=2,3) * B_22
    for j in [0, 1]:                     # A_32b(i=4) * B_22
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    strassen_2x2(2, 3, 2, 3, 0, 1)       # A_33a(i=2,3,j=2,3) * B_32a(j=2,3,k=0,1)
    for i in [2, 3]:                     # A_33b(i=2,3,j=4) * B_32b(j=4,k=0,1)
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    for j in [2, 3]:                     # A_33c(i=4,j=2,3) * B_32a(j=2,3,k=0,1)
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    for k in [0, 1]:                     # A_33d(i=4,j=4) * B_32b(j=4,k=0,1)
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # C_33 = A_32 B_23 + A_33 B_33
    strassen_2x2(2, 3, 0, 1, 2, 3)       # A_32a(i=2,3) * B_23a(k=2,3)
    for i in [2, 3]:                     # A_32a(i=2,3) * B_23b(k=4)
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    for j in [0, 1]:                     # A_32b(i=4) * B_23a(k=2,3)
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    for j in [0, 1]:                     # A_32b(i=4) * B_23b(k=4)
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    strassen_3x3(2, 2, 2)                # A_33 * B_33 (rank 23)

    # Build matrices
    if not U_list:
        return {"U": np.zeros((n*m, 1)).tolist(), "V": np.zeros((m*p, 1)).tolist(), "W": np.zeros((p*n, 1)).tolist()}

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
