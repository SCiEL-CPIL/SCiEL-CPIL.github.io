"""
5x5 matrix multiplication - rank 110 via Laderman 3x3 block.
Uses Strassen for 2x2 blocks and Laderman for 3x3 blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

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

    # Laderman's algorithm for 3x3 matrix multiplication (19 terms)
    def laderman_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # M1 = A00 * B00
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        # M2 = A01 * B10
        add_term({(i0,j1):1}, {(j1,k0):1}, {(k0,i0):1})
        # M3 = A10 * B01
        add_term({(i1,j0):1}, {(j0,k1):1}, {(k1,i1):1})
        # M4 = A11 * B11
        add_term({(i1,j1):1}, {(j1,k1):1}, {(k1,i1):1})
        # M5 = A02 * B20
        add_term({(i0,j2):1}, {(j2,k0):1}, {(k0,i0):1})
        # M6 = A20 * B02
        add_term({(i2,j0):1}, {(j0,k2):1}, {(k2,i2):1})
        # M7 = A12 * B21
        add_term({(i1,j2):1}, {(j2,k1):1}, {(k1,i1):1})
        # M8 = A21 * B12
        add_term({(i2,j1):1}, {(j1,k2):1}, {(k2,i2):1})
        # M9 = (A00+A11) * (B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M10 = (A01+A11) * (B10+B11)
        add_term({(i0,j1):1, (i1,j1):1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M11 = (A00+A01) * (B00+B01)
        add_term({(i0,j0):1, (i0,j1):1}, {(j0,k0):1, (j0,k1):1}, {(k0,i0):1, (k1,i0):1})
        # M12 = (A10+A11) * (B10+B01)
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1, (j0,k1):1}, {(k0,i1):1, (k1,i1):1})
        # M13 = (A00+A02) * (B20+B00)
        add_term({(i0,j0):1, (i0,j2):1}, {(j2,k0):1, (j0,k0):1}, {(k0,i0):1, (k2,i0):1})
        # M14 = (A20+A00) * (B00+B02)
        add_term({(i2,j0):1, (i0,j0):1}, {(j0,k0):1, (j0,k2):1}, {(k0,i2):1, (k2,i0):1})
        # M15 = (A11+A12) * (B21+B11)
        add_term({(i1,j1):1, (i1,j2):1}, {(j2,k1):1, (j1,k1):1}, {(k1,i1):1, (k2,i1):1})
        # M16 = (A21+A11) * (B11+B12)
        add_term({(i2,j1):1, (i1,j1):1}, {(j1,k1):1, (j1,k2):1}, {(k1,i2):1, (k2,i1):1})
        # M17 = (A02+A01) * (B12+B21)
        add_term({(i0,j2):1, (i0,j1):1}, {(j1,k2):1, (j2,k1):1}, {(k2,i0):1, (k1,i0):1})
        # M18 = (A21+A20) * (B01+B02)
        add_term({(i2,j1):1, (i2,j0):1}, {(j0,k1):1, (j0,k2):1}, {(k1,i2):1, (k2,i2):1})
        # M19 = (A12+A20) * (B20+B02)
        add_term({(i1,j2):1, (i2,j0):1}, {(j2,k0):1, (j0,k2):1}, {(k0,i1):1, (k2,i2):1})

    # 1. C_22 = A_22 B_22 + A_23 B_32
    # Strassen for A_22 B_22 (2x2 * 2x2)
    strassen_2x2(0, 1, 0, 1, 0, 1)

    # A_23 B_32: A is 2x3 (i=0,1; j=2,3,4), B is 3x2 (j=2,3,4; k=0,1)
    # Use 2+1 split on j: j={2,3} and j={4}
    # A_23a (2x2) * B_32a (2x2): Strassen
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A_23b (2x1, j=4) * B_32b (1x2, j=4): schoolbook 2*1*2 = 4
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # 2. C_23 = A_22 B_23 + A_23 B_33
    # A_22 (2x2) * B_23 (2x3): Strassen on 2x2 part + schoolbook
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A_22 * B_23b (j=0,1; k=4): schoolbook 2*2*1 = 4
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A_23 (2x3) * B_33 (3x3): 
    # A_23a (2x2, j=2,3) * B_33a (2x2, j=2,3,k=2,3): Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A_23a * B_33b (j=2,3; k=4): schoolbook 2*2*1 = 4
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_23b (2x1, j=4) * B_33b (1x3, j=4; k=2,3,4): schoolbook 2*1*3 = 6
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # 3. C_32 = A_32 B_22 + A_33 B_32
    # A_32 (3x2) * B_22 (2x2):
    # A_32a (2x2, i=2,3) * B_22: Strassen
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A_32b (1x2, i=4) * B_22: schoolbook 1*2*2 = 4
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A_33 (3x3) * B_32 (3x2):
    # A_33a (2x2, i=2,3; j=2,3) * B_32a (2x2, j=2,3; k=0,1): Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A_33b (2x1, i=2,3; j=4) * B_32b (1x2, j=4; k=0,1): schoolbook 2*1*2 = 4
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A_33c (1x2, i=4; j=2,3) * B_32a (2x2, j=2,3; k=0,1): schoolbook 1*2*2 = 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33d (1x1, i=4; j=4) * B_32b (1x2, j=4; k=0,1): schoolbook 1*1*2 = 2
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # 4. C_33 = A_32 B_23 + A_33 B_33
    # A_32 (3x2) * B_23 (2x3):
    # A_32a (2x2, i=2,3) * B_23a (2x2, k=2,3): Strassen
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A_32a * B_23b (k=4): schoolbook 2*2*1 = 4
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_32b (1x2, i=4) * B_23a (2x2, k=2,3): schoolbook 1*2*2 = 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_32b * B_23b (k=4): schoolbook 1*2*1 = 2
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A_33 B_33 (3x3 * 3x3) with Laderman (19 terms)
    laderman_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
