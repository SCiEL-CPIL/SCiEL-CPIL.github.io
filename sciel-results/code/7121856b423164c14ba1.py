"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Kronecker product decomposition
    # M_{2,4,8} = M_{1,2,4} ⊗ M_{2,2,2}
    # 
    # M_{1,2,4}: multiply 1x2 by 2x4. Rank = 4 (schoolbook, minimal).
    #   U_A: (1*2, 4) = (2, 4),  V_A: (2*4, 4) = (8, 4),  W_A: (4*1, 4) = (4, 4)
    #   Term r (r=0..3): U_A[j, r] = 1 if j==r else 0
    #                    V_A[j*4+k, r] = 1 if j==r else 0
    #                    W_A[k, r] = 1 for all k
    #
    # M_{2,2,2}: multiply 2x2 by 2x2. Rank = 7 (Strassen).
    #   U_B: (2*2, 7) = (4, 7),  V_B: (2*2, 7) = (4, 7),  W_B: (2*2, 7) = (4, 7)
    #   Strassen terms:
    #     M0 = (A00+A11)(B00+B11)
    #     M1 = (A10+A11)B00
    #     M2 = A00(B10-B11)
    #     M3 = A11(B01-B00)
    #     M4 = (A00+A01)B11
    #     M5 = (A10-A00)(B00+B10)
    #     M6 = (A01-A11)(B01+B11)
    #
    # Combined: rank = 4 * 7 = 28
    # U_combined: (n*m, 28) = (8, 28)
    # V_combined: (m*p, 28) = (32, 28)
    # W_combined: (p*n, 28) = (16, 28)
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M_{1,2,4} decomposition (rank 4)
    # i_A = 0 always (only 1 row in A)
    # j_A in {0, 1} (2 columns in A, 2 rows in B)
    # k_A in {0, 1, 2, 3} (4 columns in B)
    # U_A[j_A, r_A] = 1 if j_A == r_A else 0
    # V_A[j_A * 4 + k_A, r_A] = 1 if j_A == r_A else 0
    # W_A[k_A, r_A] = 1
    
    # M_{2,2,2} Strassen decomposition (rank 7)
    # i_B in {0, 1}, j_B in {0, 1}, k_B in {0, 1}
    # U_B[i_B * 2 + j_B, r_B], V_B[j_B * 2 + k_B, r_B], W_B[k_B * 2 + i_B, r_B]
    
    # Strassen coefficients
    # Term 0: M0 = (A00+A11)(B00+B11)
    #   U: A00=1, A11=1, A10=0, A01=0  -> [1, 0, 0, 1] for (i,j) = (0,0),(1,0),(0,1),(1,1)
    #   V: B00=1, B11=1, B10=0, B01=0  -> [1, 0, 0, 1] for (j,k) = (0,0),(1,0),(0,1),(1,1)
    #   W: C gets M0 at C00 and C11, subtracts at C01 and C10
    #     C00 += M0, C01 -= M0, C10 -= M0, C11 += M0
    #     W[k*2+i]: k=0,i=0 -> +1; k=0,i=1 -> -1; k=1,i=0 -> -1; k=1,i=1 -> +1
    #     -> [1, -1, -1, 1]
    
    # Term 1: M1 = (A10+A11)B00
    #   U: A10=1, A11=1 -> [0, 1, 0, 1]
    #   V: B00=1 -> [1, 0, 0, 0]
    #   W: C10 += M1 -> k=1,i=0: +1 -> [0, 0, 1, 0]
    
    # Term 2: M2 = A00(B10-B11)
    #   U: A00=1 -> [1, 0, 0, 0]
    #   V: B10=1, B11=-1 -> [0, 1, 0, -1]
    #   W: C10 += M2 -> k=1,i=0: +1 -> [0, 0, 1, 0]
    
    # Term 3: M3 = A11(B01-B00)
    #   U: A11=1 -> [0, 0, 0, 1]
    #   V: B01=1, B00=-1 -> [0, 1, -1, 0]
    #   W: C01 += M3 -> k=0,i=1: +1 -> [0, 1, 0, 0]
    
    # Term 4: M4 = (A00+A01)B11
    #   U: A00=1, A01=1 -> [1, 0, 1, 0]
    #   V: B11=1 -> [0, 0, 0, 1]
    #   W: C11 += M4 -> k=1,i=1: +1 -> [0, 0, 0, 1]
    
    # Term 5: M5 = (A10-A00)(B00+B10)
    #   U: A10=1, A00=-1 -> [0, 1, -1, 0]
    #   V: B00=1, B10=1 -> [1, 0, 1, 0]
    #   W: C01 += M5 -> k=0,i=1: +1 -> [0, 1, 0, 0]
    
    # Term 6: M6 = (A01-A11)(B01+B11)
    #   U: A01=1, A11=-1 -> [0, 0, 1, -1]
    #   V: B01=1, B11=1 -> [0, 1, 0, 1]
    #   W: C11 += M6 -> k=1,i=1: +1 -> [0, 0, 0, 1]
    
    U_B = np.array([
        [1,  0,  1,  0,  1,  0,  0],   # i_B=0, j_B=0
        [0,  1,  0,  0,  0,  1,  0],   # i_B=1, j_B=0
        [0,  0,  0,  0,  1, -1,  1],   # i_B=0, j_B=1
        [1,  1,  0,  1,  0,  0, -1],   # i_B=1, j_B=1
    ], dtype=float)
    
    V_B = np.array([
        [1,  1,  0,  0,  0,  1,  0],   # j_B=0, k_B=0
        [0,  0,  1,  1,  0,  0,  1],   # j_B=1, k_B=0
        [0,  0,  0, -1,  0,  1,  0],   # j_B=0, k_B=1
        [1,  0, -1,  0,  1,  0,  1],   # j_B=1, k_B=1
    ], dtype=float)
    
    W_B = np.array([
        [1,  0,  0,  0,  0,  0,  0],   # k_B=0, i_B=0
        [-1, 0, 0, 1, 0, 1, 0],        # k_B=0, i_B=1
        [-1, 1, 1, 0, 0, 0, 0],        # k_B=1, i_B=0
        [1,  0, 0, 0, 1, 0, 1],        # k_B=1, i_B=1
    ], dtype=float)
    
    # Verify Strassen decomposition
    # C[i_B, k_B] = sum_{j_B} A[i_B, j_B] * B[j_B, k_B]
    for i_B in range(2):
        for k_B in range(2):
            expected = 0
            for j_B in range(2):
                expected += 1 * 1  # all entries are 1 in the tensor
            # Actually the tensor M_{2,2,2} has T[i,j,k] = 1 for all i,j,k
            # So expected sum over j_B of U_B[i_B*2+j_B, r] * V_B[j_B*2+k_B, r] * W_B[k_B*2+i_B, r] should be 1
            actual = 0
            for r_B in range(7):
                for j_B in range(2):
                    actual += U_B[i_B * 2 + j_B, r_B] * V_B[j_B * 2 + k_B, r_B] * W_B[k_B * 2 + i_B, r_B]
            if abs(actual - 1) > 1e-10:
                # Fallback: use schoolbook for M_{2,2,2}
                pass
    
    # Now build the combined decomposition
    # Index mapping:
    # i = i_B (i_A = 0)
    # j = 2*j_A + j_B
    # k = 2*k_A + k_B
    # U index: i * 4 + j = i_B * 4 + 2*j_A + j_B
    # V index: j * 8 + k = (2*j_A + j_B) * 8 + (2*k_A + k_B)
    # W index: k * 2 + i = (2*k_A + k_B) * 2 + i_B
    
    for r_A in range(4):
        for r_B in range(7):
            r = r_A * 7 + r_B
            for j_A in range(2):
                for i_B in range(2):
                    for j_B in range(2):
                        # U contribution
                        u_val = (1.0 if j_A == r_A else 0.0) * U_B[i_B * 2 + j_B, r_B]
                        if u_val != 0:
                            U[i_B * 4 + 2 * j_A + j_B, r] += u_val
                        
                        for k_A in range(4):
                            for k_B in range(2):
                                # V contribution
                                v_val = (1.0 if j_A == r_A else 0.0) * V_B[j_B * 2 + k_B, r_B]
                                if v_val != 0:
                                    V[(2 * j_A + j_B) * 8 + 2 * k_A + k_B, r] += v_val
                                
                                # W contribution
                                w_val = 1.0 * W_B[k_B * 2 + i_B, r_B]
                                if w_val != 0:
                                    W[(2 * k_A + k_B) * 2 + i_B, r] += w_val
    
    # Verify the full decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i * m + j, idx] = 1
                                V[j * p + k, idx] = 1
                                W[k * n + i, idx] = 1
                                idx += 1
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
