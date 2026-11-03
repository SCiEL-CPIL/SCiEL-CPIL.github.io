"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # We express 4x4x6 as Kronecker product of 2x2x3 (rank 11) and 2x2x2 (rank 7)
    # Total rank = 11 * 7 = 77
    # But we can try a different factorization: 4x4x6 = 2x2x2 ⊗ 2x2x3
    # Actually, let's try to use a lower-rank decomposition for 2x2x3.
    # Known: 2x2x3 matrix multiplication has rank at most 11 (trivial: 2*3=6 for each of 2 rows... no)
    # Actually 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    # Trivial rank = 2*2*3 = 12.
    # Strassen-like: treat as 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11? 
    # Wait, 2x2x1: A 2x2, B 2x1, C 2x1. Rank = 4 (each output element needs 2 products, 2 outputs = 4)
    # So 7 + 4 = 11. That's what the current code does.
    
    # Alternative: Can we do 2x2x3 in rank 10 or less?
    # The tensor is M_{2,2,3}. Known lower bounds: rank >= 10? 
    # Actually, for M_{2,2,3}, the rank is known to be 11 (no improvement over trivial 12 minus 1? No, 12 is trivial, 11 is the best known)
    # Wait, let me reconsider. M_{2,2,3} = M_{2,2,2} + M_{2,2,1} in terms of tensor addition?
    # No, it's not a direct sum. The 2x2x3 tensor is NOT the direct sum of 2x2x2 and 2x2x1.
    # It's a single tensor where B has 3 columns.
    
    # Let me just verify the current approach gives rank 77 and see if there's a way to reduce it.
    # Actually, the directive says to use "Kronecker product of optimized rectangular factors" with 
    # 2x2x3 (rank 5) and 2x2x2 (rank 7) to get rank 35.
    # But rank 5 for 2x2x3 is NOT possible. The trivial rank is 12, and best known is 11.
    # So that directive is based on incorrect assumptions.
    
    # Let me instead try to see if we can use a different factorization.
    # 4x4x6: Can we write it as 2x4x3 ⊗ 2x1x2? No, dimensions don't work.
    # 4x4x6 = (2*2) x (4) x (3*2)? No, the middle dimension must be consistent.
    
    # Actually, for Kronecker product of matrix multiplication tensors:
    # M_{n1,m1,p1} ⊗ M_{n2,m2,p2} = M_{n1*n2, m1*m2, p1*p2}
    # So we need n1*n2=4, m1*m2=4, p1*p2=6.
    # Options:
    # (2,2,2) ⊗ (2,2,3): n=4, m=4, p=6 ✓
    # (2,2,3) ⊗ (2,2,2): same
    # (1,4,6) ⊗ (4,1,1): trivial, rank = 24*4 = 96
    # (2,4,3) ⊗ (2,1,2): m=4*1=4, but p=3*2=6, n=2*2=4 ✓
    #   M_{2,4,3} ⊗ M_{2,1,2}
    #   rank(M_{2,4,3}) * rank(M_{2,1,2})
    #   M_{2,1,2}: A 2x1, B 1x2, C 2x2. Rank = 4 (each of 4 outputs needs 1 product)
    #   M_{2,4,3}: A 2x4, B 4x3, C 2x3. Trivial rank = 2*4*3=24.
    #   Total = 24*4 = 96. Worse.
    
    # (4,2,3) ⊗ (1,2,2): n=4*1=4, m=2*2=4, p=3*2=6 ✓
    #   M_{4,2,3} ⊗ M_{1,2,2}
    #   M_{1,2,2}: A 1x2, B 2x2, C 1x2. Rank = 4 (each of 2 outputs needs 2 products)
    #   M_{4,2,3}: A 4x2, B 2x3, C 4x3. Trivial rank = 4*2*3=24.
    #   Total = 24*4 = 96. Worse.
    
    # So the best factorization is (2,2,2) ⊗ (2,2,3) with ranks 7 and 11, total 77.
    
    # Let me just implement the current approach cleanly and correctly.
    
    # Strassen decomposition for 2x2x2 (rank 7):
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = A*B
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing: A[i,j] with i,j in 0..1, B[j,k] with j,k in 0..1, C[i,k] with i,k in 0..1
    # U rows: i*m+j = i*2+j, V rows: j*p+k = j*2+k, W rows: k*n+i = k*2+i
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # 2x2x3 decomposition (rank 11):
    # A is 2x2, B is 2x3, C is 2x3
    # First 2 columns: use Strassen (7 terms)
    # 3rd column: trivial (4 terms)
    
    U2 = [[0]*11 for _ in range(4)]
    V2 = [[0]*11 for _ in range(6)]
    W2 = [[0]*11 for _ in range(6)]
    
    # Strassen part for columns 0-1
    for c in range(7):
        for r in range(4):
            U2[r][c] = U1[r][c]
        V2[0][c] = V1[0][c]
        V2[1][c] = V1[1][c]
        V2[3][c] = V1[2][c]
        V2[4][c] = V1[3][c]
        V2[2][c] = 0
        V2[5][c] = 0
        W2[0][c] = W1[0][c]
        W2[1][c] = W1[1][c]
        W2[2][c] = W1[2][c]
        W2[3][c] = W1[3][c]
        W2[4][c] = 0
        W2[5][c] = 0
    
    # Trivial part for column 2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U2[0][7] = 1; V2[2][7] = 1; W2[4][7] = 1
    U2[1][8] = 1; V2[5][8] = 1; W2[4][8] = 1
    U2[2][9] = 1; V2[2][9] = 1; W2[5][9] = 1
    U2[3][10] = 1; V2[5][10] = 1; W2[5][10] = 1
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # A[i,j] where i=i1*2+i2, j=j1*2+j2
    # B[j,k] where j=j1*2+j2, k=k1*3+k2
    # C[i,k] where i=i1*2+i2, k=k1*3+k2
    # U_full[i*4+j, c1*11+c2] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    # V_full[j*6+k, c1*11+c2] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    # W_full[k*4+i, c1*11+c2] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    rank = 7 * 11
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
