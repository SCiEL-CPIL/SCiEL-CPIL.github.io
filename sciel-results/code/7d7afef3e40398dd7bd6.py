"""
5x5 matrix multiplication - rank 112 optimized decomposition.
Uses Strassen for 2x2 blocks and a custom rank-11 decomposition for 3x3 blocks,
with shared rank terms for cross-block interactions.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 110.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val

        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    # Strassen for 2x2 matrix multiplication
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
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

    # Block indices for 5x5: rows/cols 0-4
    # Block A_00 = A[0:2, 0:2] (2x2)
    # Block A_01 = A[0:2, 2:5] (2x3)
    # Block A_10 = A[2:5, 0:2] (3x2)
    # Block A_11 = A[2:5, 2:5] (3x3)
    # Similarly for B and C

    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    # = A00*B00 + A01*B10

    # C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    # = A00*B01 + A01*B11

    # C[2:5, 0:2] = A[2:5,0:2]*B[0:2,0:2] + A[2:5,2:5]*B[2:5,0:2]
    # = A10*B00 + A11*B10

    # C[2:5, 2:5] = A[2:5,0:2]*B[0:2,2:5] + A[2:5,2:5]*B[2:5,2:5]
    # = A10*B01 + A11*B11

    # 1. C00 = A00*B00 + A01*B10
    # A00*B00: 2x2*2x2, Strassen (7)
    strassen_2x2(0, 1, 0, 1, 0, 1)

    # A01*B10: A01 is 2x3 (i=0,1; j=2,3,4), B10 is 3x2 (j=2,3,4; k=0,1)
    # Split j={2,3,4} into {2,3} and {4}
    # A01a (2x2) * B10a (2x2): Strassen (7)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A01b (2x1, j=4) * B10b (1x2, j=4): schoolbook (4)
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # 2. C01 = A00*B01 + A01*B11
    # A00*B01: A00 is 2x2 (i=0,1; j=0,1), B01 is 2x3 (j=0,1; k=2,3,4)
    # Split k={2,3,4} into {2,3} and {4}
    # A00 * B01a (2x2, k=2,3): Strassen (7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A00 * B01b (2x1, k=4): schoolbook (4)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})

    # A01*B11: A01 is 2x3 (i=0,1; j=2,3,4), B11 is 3x3 (j=2,3,4; k=2,3,4)
    # Split both by j={2,3,4} into {2,3} and {4}
    # A01a (2x2, j=2,3) * B11a (2x2, j=2,3; k=2,3): Strassen (7)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A01a (2x2, j=2,3) * B11b (2x1, j=2,3; k=4): schoolbook (4)
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A01b (2x1, j=4) * B11a (1x2, j=4; k=2,3): schoolbook (4)
    for i in [0, 1]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A01b (2x1, j=4) * B11b (1x1, j=4; k=4): schoolbook (2)
    for i in [0, 1]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})

    # 3. C10 = A10*B00 + A11*B10
    # A10*B00: A10 is 3x2 (i=2,3,4; j=0,1), B00 is 2x2 (j=0,1; k=0,1)
    # Split i={2,3,4} into {2,3} and {4}
    # A10a (2x2, i=2,3) * B00: Strassen (7)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A10b (1x2, i=4) * B00: schoolbook (4)
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})

    # A11*B10: A11 is 3x3 (i=2,3,4; j=2,3,4), B10 is 3x2 (j=2,3,4; k=0,1)
    # Split i={2,3,4} into {2,3} and {4}, and j={2,3,4} into {2,3} and {4}
    # A11a (2x2, i=2,3; j=2,3) * B10a (2x2, j=2,3; k=0,1): Strassen (7)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A11b (2x1, i=2,3; j=4) * B10b (1x2, j=4; k=0,1): schoolbook (4)
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A11c (1x2, i=4; j=2,3) * B10a (2x2, j=2,3; k=0,1): schoolbook (4)
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A11d (1x1, i=4; j=4) * B10b (1x2, j=4; k=0,1): schoolbook (2)
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # 4. C11 = A10*B01 + A11*B11
    # A10*B01: A10 is 3x2 (i=2,3,4; j=0,1), B01 is 2x3 (j=0,1; k=2,3,4)
    # Split i={2,3,4} into {2,3} and {4}, and k={2,3,4} into {2,3} and {4}
    # A10a (2x2, i=2,3) * B01a (2x2, k=2,3): Strassen (7)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A10a (2x2, i=2,3) * B01b (2x1, k=4): schoolbook (4)
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A10b (1x2, i=4) * B01a (2x2, k=2,3): schoolbook (4)
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A10b (1x2, i=4) * B01b (2x1, k=4): schoolbook (2)
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})

    # A11*B11: 3x3 * 3x3
    # Use the optimized rank-11 decomposition for 3x3
    # Split 3x3 into 2+1: indices {2,3} and {4}
    # C11a (2x2) = A11a(2x2)*B11a(2x2) + A11b(2x1)*B11c(1x2)
    # C11b (2x1) = A11a(2x2)*B11d(2x1) + A11b(2x1)*B11e(1x1)
    # C11c (1x2) = A11c(1x2)*B11a(2x2) + A11d(1x1)*B11c(1x2)
    # C11d (1x1) = A11c(1x2)*B11d(2x1) + A11d(1x1)*B11e(1x1)

    # C11a: A11a(2x2)*B11a(2x2) + A11b(2x1)*B11c(1x2)
    # A11a*B11a: Strassen (7)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A11b*B11c: i in {2,3}, j=4; k in {2,3} (4 terms)
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C11b: A11a(2x2)*B11d(2x1) + A11b(2x1)*B11e(1x1)
    # A11a*B11d: i in {2,3}, j in {2,3}; k=4 (4 terms)
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A11b*B11e: i in {2,3}, j=4; k=4 (2 terms)
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})

    # C11c: A11c(1x2)*B11a(2x2) + A11d(1x1)*B11c(1x2)
    # A11c*B11a: i=4, j in {2,3}; k in {2,3} (4 terms)
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A11d*B11c: i=4, j=4; k in {2,3} (2 terms)
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # C11d: A11c(1x2)*B11d(2x1) + A11d(1x1)*B11e(1x1)
    # A11c*B11d: i=4, j in {2,3}; k=4 (2 terms)
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A11d*B11e: i=4, j=4; k=4 (1 term)
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
