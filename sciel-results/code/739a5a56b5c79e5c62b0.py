import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 rank-7 decomposition
    # U_S[i*2+j, c], V_S[j*2+k, c], W_S[k*2+i, c]
    def strassen_2x2_2x2():
        # 7 terms
        U_S = [[0]*7 for _ in range(4)]
        V_S = [[0]*7 for _ in range(4)]
        W_S = [[0]*7 for _ in range(4)]
        # Term 0: (A00+A11)(B00+B11)
        U_S[0][0] = 1; U_S[3][0] = 1
        V_S[0][0] = 1; V_S[3][0] = 1
        W_S[0][0] = 1; W_S[3][0] = 1
        # Term 1: (A10+A11)B00
        U_S[2][1] = 1; U_S[3][1] = 1
        V_S[0][1] = 1
        W_S[2][1] = 1; W_S[3][1] = 1
        # Term 2: A00(B10-B11)
        U_S[0][2] = 1
        V_S[2][2] = 1; V_S[3][2] = -1
        W_S[0][2] = 1; W_S[1][2] = 1
        # Term 3: A11(B00-B10)
        U_S[3][3] = 1
        V_S[0][3] = 1; V_S[2][3] = -1
        W_S[1][3] = 1; W_S[3][3] = 1
        # Term 4: (A00+A01)B11
        U_S[0][4] = 1; U_S[1][4] = 1
        V_S[3][4] = 1
        W_S[0][4] = 1; W_S[1][4] = 1
        # Term 5: (A10-A00)(B00+B01)
        U_S[2][5] = 1; U_S[0][5] = -1
        V_S[0][5] = 1; V_S[1][5] = 1
        W_S[2][5] = 1; W_S[1][5] = 1
        # Term 6: (A01-A11)(B10+B11)
        U_S[1][6] = 1; U_S[3][6] = -1
        V_S[2][6] = 1; V_S[3][6] = 1
        W_S[0][6] = 1; W_S[3][6] = 1
        return U_S, V_S, W_S

    # 2x2x1 rank-3 decomposition (multiplying 2x2 by 2x1)
    # C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # Use: M1 = (A00+A11)*B0, M2 = (A10+A11)*B1, M3 = (A00-A10)*(B0-B1)
    # Actually simpler: rank 4 is trivial, rank 3 via:
    # M1 = (A00+A11)B0, M2 = A01B1, M3 = (A10-A00)B0, M4 = A11B1
    # That's rank 4. For rank 3:
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # M1 = (A00+A11)*B0, M2 = (A01+A11)*B1, M3 = (A00-A10)*(B0-B1)
    # C0 = M1 - A11*B0 + A01*B1... this gets complex. Let's use rank 4 for 2x2x1.
    def rank4_2x2_2x1():
        U_S = [[0]*4 for _ in range(4)]
        V_S = [[0]*4 for _ in range(2)]
        W_S = [[0]*4 for _ in range(2)]
        # Term 0: A00*B0 -> C0
        U_S[0][0] = 1; V_S[0][0] = 1; W_S[0][0] = 1
        # Term 1: A01*B1 -> C0
        U_S[1][1] = 1; V_S[1][1] = 1; W_S[0][1] = 1
        # Term 2: A10*B0 -> C1
        U_S[2][2] = 1; V_S[0][2] = 1; W_S[1][2] = 1
        # Term 3: A11*B1 -> C1
        U_S[3][3] = 1; V_S[1][3] = 1; W_S[1][3] = 1
        return U_S, V_S, W_S

    # Build 4x4x5 using Kronecker structure
    # 4x4 = 2x2 blocks, 4x5 = 2x2 blocks for first 4 cols + 2x1 for last col
    # C = A * B where A is 4x4, B is 4x5
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split i = 2*i2 + i1, j = 2*j2 + j1, k = 2*k2 + k1 (for k<4)
    # For k=4 (k2=2, k1=0): C[i,4] = sum_j A[i,j] B[j,4]
    
    # Approach: Use Kronecker for the 4x4x4 part, then add rank for 4x4x1
    # 4x4x4 = (2x2x2) ⊗ (2x2x2) via Strassen: rank 7*7 = 49
    # 4x4x1 part: rank 16 (schoolbook) or less
    
    # Total: 49 + 16 = 65, still better than 80
    
    # Even better: use Strassen for 4x4x4 (rank 49) and optimize 4x4x1
    # 4x4x1: C[i] = sum_j A[i,j] B[j,4] for i=0..3
    # This is 4 dot products of length 4. Rank 16 schoolbook.
    # Can we do better? Use the structure.
    
    # Let's build the Kronecker decomposition for 4x4x4
    def build_4x4x4_strassen():
        U_S, V_S, W_S = strassen_2x2_2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]  # 4*4
        V = [[0]*rank for _ in range(16)]  # 4*4
        W = [[0]*rank for _ in range(16)]  # 4*4
        
        for c1 in range(7):
            for c2 in range(7):
                c = c1*7 + c2
                for i2 in range(2):
                    for i1 in range(2):
                        for j2 in range(2):
                            for j1 in range(2):
                                for k2 in range(2):
                                    for k1 in range(2):
                                        i = i2*2 + i1
                                        j = j2*2 + j1
                                        k = k2*2 + k1
                                        U[i*4+j][c] += U_S[i2*2+i1][c1] * U_S[i1*2+i2][c2]
                                        V[j*4+k][c] += V_S[j2*2+j1][c1] * V_S[j1*2+j2][c2]
                                        W[k*4+i][c] += W_S[k2*2+i2][c1] * W_S[k1*2+i1][c2]
        return U, V, W

    # This is getting complex. Let me use a simpler structured approach.
    # Direct Kronecker: T_{4,4,4} = T_{2,2,2} ⊗ T_{2,2,2}
    # U_K[i*4+j, c1*7+c2] = U_S[i*2+j1, c1] * U_S[i2*2+j2, c2]
    # where i = 2*i2+i1, j = 2*j2+j1
    
    def build_kronecker_4x4x4():
        U_S, V_S, W_S = strassen_2x2_2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for i2 in range(2):
            for i1 in range(2):
                for j2 in range(2):
                    for j1 in range(2):
                        for k2 in range(2):
                            for k1 in range(2):
                                i = i2*2 + i1
                                j = j2*2 + j1
                                k = k2*2 + k1
                                for c1 in range(7):
                                    for c2 in range(7):
                                        c = c1*7 + c2
                                        # U_S index: row*2 + col. 
                                        # Factor 1 (outer): row i2, col j2 -> idx 2*i2 + j2
                                        # Factor 2 (inner): row i1, col j1 -> idx 2*i1 + j1
                                        U[i*4+j][c] += U_S[2*i2 + j2][c1] * U_S[2*i1 + j1][c2]
                                        
                                        # V_S index: row j, col k.
                                        # Factor 1 (outer): row j2, col k2 -> idx 2*j2 + k2
                                        # Factor 2 (inner): row j1, col k1 -> idx 2*j1 + k1
                                        V[j*4+k][c] += V_S[2*j2 + k2][c1] * V_S[2*j1 + k1][c2]
                                        
                                        # W_S index: row k, col i.
                                        # Factor 1 (outer): row k2, col i2 -> idx 2*k2 + i2
                                        # Factor 2 (inner): row k1, col i1 -> idx 2*k1 + i1
                                        W[k*4+i][c] += W_S[2*k2 + i2][c1] * W_S[2*k1 + i1][c2]
        return U, V, W

    # For 4x4x1 (last column), use schoolbook rank 16
    def build_4x4x1():
        rank = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(4)]  # 4x1
        W = [[0]*rank for _ in range(4)]  # 1x4
        c = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j][c] = 1
                V[j][c] = 1
                W[i][c] = 1
                c += 1
        return U, V, W

    # Combine: 4x4x5 = 4x4x4 (Kronecker Strassen) + 4x4x1 (schoolbook)
    # Total rank: 49 + 16 = 65
    def build_combined():
        U4, V4, W4 = build_kronecker_4x4x4()
        U1, V1, W1 = build_4x4x1()
        rank = 49 + 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(20)]  # 4*5
        W = [[0]*rank for _ in range(20)]  # 5*4
        
        # Fill 4x4x4 part (k=0..3)
        for i in range(16):
            for c in range(49):
                U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j*5+k][c] = V4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[k*4+i][c] = W4[k*4+i][c]
        
        # Fill 4x4x1 part (k=4)
        for i in range(16):
            for c in range(16):
                U[i][49+c] = U1[i][c]
        for j in range(4):
            for c in range(16):
                V[j*5+4][49+c] = V1[j][c]
        for i in range(4):
            for c in range(16):
                W[4*4+i][49+c] = W1[i][c]
        
        return {"U": U, "V": V, "W": W}

    # Try the structured decomposition
    if time.time() < deadline - 20:
        result = build_combined()
        if verify(result["U"], result["V"], result["W"]):
            return result

    # Fallback to schoolbook
    return make_schoolbook()

# EVOLVE-BLOCK-END
