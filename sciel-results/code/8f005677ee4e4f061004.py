import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0  # Safety margin
    
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

    # Helper for Strassen 2x2
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

    # Helper for 2x2 * 2x1 (Rank 4)
    def mul_2x2_2x1(i0, i1, j0, j1, k0):
        # C0 = A00 B0 + A10 B1
        # C1 = A01 B0 + A11 B1
        # Terms:
        # 1. A00 B0 -> C0
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        # 2. A10 B0 -> C0
        add_term({(i1,j0):1}, {(j0,k0):1}, {(k0,i1):1})
        # 3. A01 B0 -> C1
        add_term({(i0,j1):1}, {(j1,k0):1}, {(k0,i0):1})
        # 4. A11 B0 -> C1
        add_term({(i1,j1):1}, {(j1,k0):1}, {(k0,i1):1})

    # Helper for 1x2 * 2x2 (Rank 4)
    def mul_1x2_2x2(i0, j0, j1, k0, k1):
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # Terms:
        # 1. A0 B00 -> C0
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        # 2. A1 B00 -> C0
        add_term({(i0,j1):1}, {(j1,k0):1}, {(k0,i0):1})
        # 3. A0 B01 -> C1
        add_term({(i0,j0):1}, {(j0,k1):1}, {(k1,i0):1})
        # 4. A1 B01 -> C1
        add_term({(i0,j1):1}, {(j1,k1):1}, {(k1,i0):1})

    # Helper for 1x2 * 2x1 (Rank 2)
    def mul_1x2_2x1(i0, j0, j1, k0):
        # C0 = A0 B0 + A1 B1
        # 1. A0 B0
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        # 2. A1 B1
        add_term({(i0,j1):1}, {(j1,k0):1}, {(k0,i0):1})

    # Helper for 2x1 * 1x2 (Rank 4)
    def mul_2x1_1x2(i0, i1, j0, k0, k1):
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        add_term({(i0,j0):1}, {(j0,k1):1}, {(k1,i0):1})
        add_term({(i1,j0):1}, {(j0,k0):1}, {(k0,i1):1})
        add_term({(i1,j0):1}, {(j0,k1):1}, {(k1,i1):1})

    # Helper for 2x1 * 1x1 (Rank 2)
    def mul_2x1_1x1(i0, i1, j0, k0):
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        add_term({(i1,j0):1}, {(j0,k0):1}, {(k0,i1):1})

    # Helper for 1x1 * 1x2 (Rank 2)
    def mul_1x1_1x2(i0, j0, k0, k1):
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        add_term({(i0,j0):1}, {(j0,k1):1}, {(k1,i0):1})

    # Helper for 1x1 * 1x1 (Rank 1)
    def mul_1x1_1x1(i0, j0, k0):
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})

    # Helper for 2x2 * 2x3 (Rank 11: 7+4)
    def mul_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        # C00, C01 from A00,A01 and B00,B01,B02
        # C10, C11 from A10,A11 and B00,B01,B02
        # Split k into {k0,k1} and {k2}
        # Part 1: A (2x2) * B (2x2) for k in {k0,k1}
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # Part 2: A (2x2) * B (2x1) for k=k2
        mul_2x2_2x1(i0, i1, j0, j1, k2)

    # Helper for 2x3 * 2x2 (Rank 11: 7+4)
    def mul_2x3_2x2(i0, i1, j0, j1, j2, k0, k1):
        # Split j into {j0,j1} and {j2}
        # Part 1: A (2x2) * B (2x2) for j in {j0,j1}
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # Part 2: A (2x1) * B (1x2) for j=j2
        mul_2x1_1x2(i0, i1, j2, k0, k1)

    # Helper for 2x3 * 3x3 (Rank 17: 11+6)
    def mul_2x3_3x3(i0, i1, j0, j1, j2, k0, k1, k2):
        # Split j into {j0,j1} and {j2}
        # Part 1: A (2x2) * B (2x3) for j in {j0,j1}
        mul_2x2_2x3(i0, i1, j0, j1, k0, k1, k2)
        # Part 2: A (2x1) * B (1x3) for j=j2
        # A (2x1) * B (1x3) -> Rank 6
        for i in [i0, i1]:
            for k in [k0, k1, k2]:
                add_term({(i,j2):1}, {(j2,k):1}, {(k,i):1})

    # Helper for 3x2 * 2x3 (Rank 17: 7+4+4+2)
    def mul_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        # Split i into {i0,i1} and {i2}
        # Split k into {k0,k1} and {k2}
        # Part 1: A (2x2) * B (2x2) for i in {i0,i1}, k in {k0,k1}
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # Part 2: A (2x2) * B (2x1) for i in {i0,i1}, k=k2
        mul_2x2_2x1(i0, i1, j0, j1, k2)
        # Part 3: A (1x2) * B (2x2) for i=i2, k in {k0,k1}
        mul_1x2_2x2(i2, j0, j1, k0, k1)
        # Part 4: A (1x2) * B (2x1) for i=i2, k=k2
        mul_1x2_2x1(i2, j0, j1, k2)

    # Helper for 3x3 * 3x3 (Rank 17: 7+4+4+2)
    def mul_3x3_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # Split i into {i0,i1} and {i2}
        # Split k into {k0,k1} and {k2}
        # Split j into {j0,j1} and {j2}
        
        # C_22: i in {i0,i1}, k in {k0,k1}
        # A (2x2) * B (2x2)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        
        # C_21: i in {i0,i1}, k=k2
        # A (2x2) * B (2x1)
        mul_2x2_2x1(i0, i1, j0, j1, k2)
        
        # C_12: i=i2, k in {k0,k1}
        # A (1x2) * B (2x2)
        mul_1x2_2x2(i2, j0, j1, k0, k1)
        
        # C_11: i=i2, k=k2
        # A (1x2) * B (2x1)
        mul_1x2_2x1(i2, j0, j1, k2)

    # Indices
    # A: i in 0..4, j in 0..4
    # B: j in 0..4, k in 0..4
    # C: i in 0..4, k in 0..4
    
    # Split 5 into 2+3
    # I2 = {0,1}, I3 = {2,3,4}
    # J2 = {0,1}, J3 = {2,3,4}
    # K2 = {0,1}, K3 = {2,3,4}
    
    # 1. C_22 = A_22 B_22 + A_23 B_32
    # A_22 B_22: 2x2 * 2x2
    strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # A_23 B_32: 2x3 * 3x2
    mul_2x3_2x2(0, 1, 2, 3, 4, 0, 1)
    
    # 2. C_23 = A_22 B_23 + A_23 B_33
    # A_22 B_23: 2x2 * 2x3
    mul_2x2_2x3(0, 1, 0, 1, 2, 3, 4)
    
    # A_23 B_33: 2x3 * 3x3
    mul_2x3_3x3(0, 1, 2, 3, 4, 2, 3, 4)
    
    # 3. C_32 = A_32 B_22 + A_33 B_32
    # A_32 B_22: 3x2 * 2x2
    # Symmetric to A_22 B_23?
    # A_32 (3x2) * B_22 (2x2)
    # Split i (3) into 2+1
    # Part 1: A (2x2) * B (2x2) -> Strassen
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # Part 2: A (1x2) * B (2x2) -> Rank 4
    mul_1x2_2x2(4, 0, 1, 0, 1)
    
    # A_33 B_32: 3x3 * 3x2
    # Symmetric to A_23 B_33?
    # A_33 (3x3) * B_32 (3x2)
    # Split i (3) into 2+1
    # Part 1: A (2x3) * B (3x2) -> Rank 11
    mul_2x3_2x2(2, 3, 2, 3, 4, 0, 1)
    # Part 2: A (1x3) * B (3x2) -> Rank 6
    for j in [2, 3, 4]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
            
    # 4. C_33 = A_32 B_23 + A_33 B_33
    # A_32 B_23: 3x2 * 2x3
    mul_3x2_2x3(2, 3, 4, 0, 1, 2, 3, 4)
    
    # A_33 B_33: 3x3 * 3x3
    mul_3x3_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
