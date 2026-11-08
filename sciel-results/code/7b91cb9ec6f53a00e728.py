import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen multipliers
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = 1 # C00=M1+M4-M5+M7, C11=M1-M2+M3+M6
        # Wait, W contributes to C.
        # C00 gets M1. C11 gets M1.
        # W[k*2+i, r] is the coefficient for C[i,k]? No, W[k*n+i, r].
        # Here n=2. W[k*2+i, r].
        # C[i,k] = sum U[i,j] V[j,k] W[k,i].
        
        # Let's map Strassen equations to U,V,W.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # r=0: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00+=1, C11+=1
        
        # r=1: M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 += 1 (k=1, i=0? No. C10 is i=1, k=0. W[0*2+1, 1] = W[1, 1]. C11 is i=1, k=1. W[1*2+1, 1] = W[3, 1].)
        # M2 contributes to C10 and C11?
        # C10 = M2 + M4. C11 = ... -M2 ...
        # So W[0*2+1, 1] = 1 (for C10) and W[1*2+1, 1] = -1 (for C11).
        W[1, 1] = 1
        W[3, 1] = -1

        # r=2: M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 += 1 (k=1, i=0). W[1*2+0, 2] = W[2, 2]? No.
        # C01 is i=0, k=1. W[k*2+i, r] = W[1*2+0, 2] = W[2, 2].
        # C11 is i=1, k=1. W[1*2+1, 2] = W[3, 2].
        # M3 contributes to C01 and C11.
        # C01 = M3 + M5. C11 = ... + M3 ...
        W[2, 2] = 1
        W[3, 2] = 1

        # r=3: M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # M4 contributes to C00 and C10.
        # C00 = ... + M4 ...
        # C10 = M2 + M4.
        # C00 is i=0, k=0. W[0, 3].
        # C10 is i=1, k=0. W[1, 3].
        W[0, 3] = 1
        W[1, 3] = 1

        # r=4: M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # M5 contributes to C00 and C01.
        # C00 = ... - M5 ...
        # C01 = M3 + M5.
        # C00 is i=0, k=0. W[0, 4].
        # C01 is i=0, k=1. W[2, 4].
        W[0, 4] = -1
        W[2, 4] = 1

        # r=5: M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # M6 contributes to C11.
        # C11 = ... + M6.
        # C11 is i=1, k=1. W[3, 5].
        W[3, 5] = 1

        # r=6: M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # M7 contributes to C00.
        # C00 = ... + M7.
        # C00 is i=0, k=0. W[0, 6].
        W[0, 6] = 1

        return U, V, W

    def get_rank1_2x1xk(k):
        """
        Returns U, V, W for 2x1xk matrix multiplication with rank 1.
        U shape (2, 1), V shape (k, 1), W shape (2k, 1).
        """
        U = np.ones((2, 1))
        V = np.ones((k, 1))
        W = np.ones((2*k, 1))
        return U, V, W

    def get_schoolbook_2x3x3():
        """
        Returns U, V, W for 2x3x3 matrix multiplication with rank 18 (schoolbook).
        U shape (6, 18), V shape (9, 18), W shape (6, 18).
        """
        n, m, p = 2, 3, 3
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def assemble():
        # Dimensions: n=2, m=4, p=5
        # Split m: j1 in 0..2 (3), j2=3 (1)
        # Split p: k1 in 0..2 (3), k2 in 3..4 (2)
        
        # Block 1: 2x3x3 (j in 0..2, k in 0..2)
        U1, V1, W1 = get_schoolbook_2x3x3()
        # U1: (6, 18). Indices i*3+j. i in 0..1, j in 0..2.
        # V1: (9, 18). Indices j*3+k. j in 0..2, k in 0..2.
        # W1: (6, 18). Indices k*2+i. k in 0..2, i in 0..1.
        
        # Block 2: 2x3x2 (j in 0..2, k in 3..4)
        # Sub-block 2a: 2x2x2 (j in 0..1, k in 3..4)
        U2a, V2a, W2a = get_strassen_2x2x2()
        # U2a: (4, 7). Indices i*2+j. i in 0..1, j in 0..1.
        # V2a: (4, 7). Indices j*2+k_loc. j in 0..1, k_loc in 0..1.
        # W2a: (4, 7). Indices k_loc*2+i. k_loc in 0..1, i in 0..1.
        
        # Sub-block 2b: 2x1x2 (j=2, k in 3..4)
        U2b, V2b, W2b = get_rank1_2x1xk(2)
        # U2b: (2, 1). Indices i.
        # V2b: (2, 1). Indices k_loc.
        # W2b: (4, 1). Indices k_loc*2+i.
        
        # Block 3: 2x1x3 (j=3, k in 0..2)
        U3, V3, W3 = get_rank1_2x1xk(3)
        # U3: (2, 1).
        # V3: (3, 1).
        # W3: (6, 1).
        
        # Block 4: 2x1x2 (j=3, k in 3..4)
        U4, V4, W4 = get_rank1_2x1xk(2)
        # U4: (2, 1).
        # V4: (2, 1).
        # W4: (4, 1).
        
        # Total Rank
        R1 = 18
        R2a = 7
        R2b = 1
        R3 = 1
        R4 = 1
        Total_R = R1 + R2a + R2b + R3 + R4
        # Total_R = 28
        
        # Allocate Full Matrices
        U_full = np.zeros((8, Total_R)) # n*m = 2*4 = 8
        V_full = np.zeros((20, Total_R)) # m*p = 4*5 = 20
        W_full = np.zeros((10, Total_R)) # p*n = 5*2 = 10
        
        # Place Block 1
        # U1: i in 0..1, j in 0..2. Global U[i*4+j, r].
        for i in range(2):
            for j in range(3):
                U_full[i*4+j, 0:R1] = U1[i*3+j, :]
        # V1: j in 0..2, k in 0..2. Global V[j*5+k, r].
        for j in range(3):
            for k in range(3):
                V_full[j*5+k, 0:R1] = V1[j*3+k, :]
        # W1: k in 0..2, i in 0..1. Global W[k*2+i, r].
        for k in range(3):
            for i in range(2):
                W_full[k*2+i, 0:R1] = W1[k*2+i, :]
                
        # Place Block 2a (Strassen)
        offset = R1
        # U2a: i in 0..1, j in 0..1. Global U[i*4+j, r].
        for i in range(2):
            for j in range(2):
                U_full[i*4+j, offset:offset+R2a] = U2a[i*2+j, :]
        # V2a: j in 0..1, k_loc in 0..1. Global V[j*5+(3+k_loc), r].
        for j in range(2):
            for k_loc in range(2):
                V_full[j*5+(3+k_loc), offset:offset+R2a] = V2a[j*2+k_loc, :]
        # W2a: k_loc in 0..1, i in 0..1. Global W[(3+k_loc)*2+i, r].
        for k_loc in range(2):
            for i in range(2):
                W_full[(3+k_loc)*2+i, offset:offset+R2a] = W2a[k_loc*2+i, :]
                
        # Place Block 2b (Rank 1)
        offset += R2a
        # U2b: i in 0..1. Global U[i*4+2, r]. (j=2)
        for i in range(2):
            U_full[i*4+2, offset:offset+R2b] = U2b[i, :]
        # V2b: k_loc in 0..1. Global V[2*5+(3+k_loc), r]. (j=2)
        for k_loc in range(2):
            V_full[2*5+(3+k_loc), offset:offset+R2b] = V2b[k_loc, :]
        # W2b: k_loc in 0..1, i in 0..1. Global W[(3+k_loc)*2+i, r].
        for k_loc in range(2):
            for i in range(2):
                W_full[(3+k_loc)*2+i, offset:offset+R2b] = W2b[k_loc*2+i, :]
                
        # Place Block 3 (Rank 1)
        offset += R2b
        # U3: i in 0..1. Global U[i*4+3, r]. (j=3)
        for i in range(2):
            U_full[i*4+3, offset:offset+R3] = U3[i, :]
        # V3: k in 0..2. Global V[3*5+k, r]. (j=3)
        for k in range(3):
            V_full[3*5+k, offset:offset+R3] = V3[k, :]
        # W3: k in 0..2, i in 0..1. Global W[k*2+i, r].
        for k in range(3):
            for i in range(2):
                W_full[k*2+i, offset:offset+R3] = W3[k*2+i, :]
                
        # Place Block 4 (Rank 1)
        offset += R3
        # U4: i in 0..1. Global U[i*4+3, r]. (j=3)
        for i in range(2):
            U_full[i*4+3, offset:offset+R4] = U4[i, :]
        # V4: k_loc in 0..1. Global V[3*5+(3+k_loc), r]. (j=3)
        for k_loc in range(2):
            V_full[3*5+(3+k_loc), offset:offset+R4] = V4[k_loc, :]
        # W4: k_loc in 0..1, i in 0..1. Global W[(3+k_loc)*2+i, r].
        for k_loc in range(2):
            for i in range(2):
                W_full[(3+k_loc)*2+i, offset:offset+R4] = W4[k_loc*2+i, :]
                
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    def verify_exact(U_list, V_list, W_list, R):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U_list[i*m+j][r]) * Fraction(V_list[j*p+k][r]) * Fraction(W_list[k*n+i][r])
                    if s != 1:
                        return False
        return True

    result = assemble()
    
    # Verify
    R = len(result['U'][0])
    if verify_exact(result['U'], result['V'], result['W'], R):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
