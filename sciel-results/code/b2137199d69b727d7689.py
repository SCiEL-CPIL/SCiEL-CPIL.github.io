def construct():
    n, m, p = 2, 2, 5
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # (4, 18)
    V = [[0] * rank for _ in range(m * p)]  # (10, 18)
    W = [[0] * rank for _ in range(p * n)]  # (10, 18)
    
    # Strassen 2x2 x 2x2 for first block (B columns 0,1)
    # Term 1: M1 = (A00+A11)(B00+B11), contributes to C00 and C11
    col = 0
    U[0][col] = 1; U[3][col] = 1  # A[0,0], A[1,1]
    V[0][col] = 1; V[1][col] = 1  # B[0,0], B[1,0] (j*5+k: j=0,k=0 → 0; j=1,k=0 → 5)
    # Wait, V[j*p+k]: j=0,k=0 → 0; j=0,k=1 → 1; j=1,k=0 → 5; j=1,k=1 → 6
    # B00 = B[0,0] → V[0*5+0] = V[0]
    # B11 = B[1,1] → V[1*5+1] = V[6]
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1  # C[0,0] → k*n+i = 0*2+0 = 0
    W[3][col] = 1  # C[1,1] → k*n+i = 1*2+1 = 3
    col += 1
    
    # Term 2: M2 = (A10+A11)B00, contributes to C10 and C11
    U[2][col] = 1; U[3][col] = 1  # A[1,0]=pos 1*2+0=2, A[1,1]=pos 3
    V[0][col] = 1  # B[0,0]
    W[2][col] = 1  # C[1,0] → k*n+i = 1*2+0 = 2
    W[3][col] = -1  # C[1,1]
    col += 1
    
    # Term 3: M3 = A00(B10-B11), contributes to C01 and C10
    U[0][col] = 1  # A[0,0]
    V[5][col] = 1; V[6][col] = -1  # B[1,0]=V[5], B[1,1]=V[6]
    W[1][col] = 1  # C[0,1] → k*n+i = 1*2+0 = 2... wait
    # C[k,i]: k is column index in B, i is row index in A/C
    # C[0,1] means i=0, k=1 → W position = k*n+i = 1*2+0 = 2
    # C[1,0] means i=1, k=0 → W position = k*n+i = 0*2+1 = 1
    W[1][col] = 1  # C[1,0] → i=1, k=0 → pos 1
    W[2][col] = 1  # C[0,1] → i=0, k=1 → pos 2
    col += 1
    
    # Term 4: M4 = A11(B01-B10), contributes to C00 and C10
    U[3][col] = 1  # A[1,1]
    V[1][col] = 1; V[5][col] = -1  # B[0,1]=V[1], B[1,0]=V[5]
    W[0][col] = 1  # C[0,0] → i=0,k=0 → pos 0
    W[1][col] = 1  # C[1,0] → i=1,k=0 → pos 1
    col += 1
    
    # Term 5: M5 = (A00+A01)B11, contributes to C01 and C00
    U[0][col] = 1; U[1][col] = 1  # A[0,0], A[0,1]
    V[6][col] = 1  # B[1,1]
    W[2][col] = 1  # C[0,1] → i=0,k=1 → pos 2
    W[0][col] = -1  # C[0,0] → i=0,k=0 → pos 0
    col += 1
    
    # Term 6: M6 = (A10-A00)(B00+B10), contributes to C11
    U[2][col] = 1; U[0][col] = -1  # A[1,0], A[0,0]
    V[0][col] = 1; V[5][col] = 1  # B[0,0], B[1,0]
    W[3][col] = 1  # C[1,1] → i=1,k=1 → pos 3
    col += 1
    
    # Term 7: M7 = (A01-A11)(B01+B11), contributes to C00 and C01
    U[1][col] = 1; U[3][col] = -1  # A[0,1], A[1,1]
    V[1][col] = 1; V[6][col] = 1  # B[0,1], B[1,1]
    W[0][col] = 1  # C[0,0]
    W[2][col] = 1  # C[0,1]
    col += 1
    
    # Second Strassen block for B columns 2,3
    # Same A indices, B indices shifted by 2 in k, C indices shifted by 2 in k
    # B[j, k] for k=2,3: V[j*5+k]
    # C[k, i] for k=2,3: W[k*2+i]
    
    # Term 8: M1' = (A00+A11)(B02+B13)
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[7][col] = 1  # B[0,2]=V[2], B[1,3]=V[7]
    W[4][col] = 1  # C[2,0] → i=0,k=2 → pos 4
    W[7][col] = 1  # C[3,1] → i=1,k=3 → pos 7
    col += 1
    
    # Term 9: M2' = (A10+A11)B02
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1  # B[0,2]
    W[6][col] = 1  # C[2,1] → i=1,k=2 → pos 6
    W[7][col] = -1  # C[3,1]
    col += 1
    
    # Term 10: M3' = A00(B12-B13)
    U[0][col] = 1
    V[7][col] = 1; V[8][col] = -1  # B[1,2]=V[7], B[1,3]=V[8]
    W[5][col] = 1  # C[2,1] → i=1,k=2 → pos 6... wait
    # C[k,i]: i is row of C (and A), k is column of C (and B)
    # C[2,1] means k=2, i=1 → W pos = k*n+i = 2*2+1 = 5
    # C[3,0] means k=3, i=0 → W pos = 3*2+0 = 6
    W[5][col] = 1  # C[2,1]
    W[6][col] = 1  # C[3,0]
    col += 1
    
    # Term 11: M4' = A11(B03-B12)
    U[3][col] = 1
    V[3][col] = 1; V[7][col] = -1  # B[0,3]=V[3], B[1,2]=V[7]
    W[4][col] = 1  # C[2,0] → i=0,k=2 → pos 4
    W[5][col] = 1  # C[2,1] → i=1,k=2 → pos 5
    col += 1
    
    # Term 12: M5' = (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1  # B[1,3]
    W[6][col] = 1  # C[3,0] → i=0,k=3 → pos 6
    W[4][col] = -1  # C[2,0] → i=0,k=2 → pos 4
    col += 1
    
    # Term 13: M6' = (A10-A00)(B02+B12)
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[7][col] = 1  # B[0,2], B[1,2]
    W[7][col] = 1  # C[3,1] → i=1,k=3 → pos 7
    col += 1
    
    # Term 14: M7' = (A01-A11)(B03+B13)
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[8][col] = 1  # B[0,3], B[1,3]
    W[4][col] = 1  # C[2,0]
    W[6][col] = 1  # C[3,0]
    col += 1
    
    # Schoolbook for 2x2 x 2x1 (B column 4)
    # C[4,i] = sum_j A[i,j] B[j,4]
    # 4 terms: (i,j,k=4) for i in {0,1}, j in {0,1}
    
    # Term 15: A[0,0]*B[0,4]
    U[0][col] = 1
    V[4][col] = 1  # B[0,4] → V[0*5+4] = V[4]
    W[8][col] = 1  # C[4,0] → i=0,k=4 → pos 4*2+0 = 8
    col += 1
    
    # Term 16: A[0,1]*B[1,4]
    U[1][col] = 1
    V[9][col] = 1  # B[1,4] → V[1*5+4] = V[9]
    W[8][col] = 1
    col += 1
    
    # Term 17: A[1,0]*B[0,4]
    U[2][col] = 1
    V[4][col] = 1
    W[9][col] = 1  # C[4,1] → i=1,k=4 → pos 4*2+1 = 9
    col += 1
    
    # Term 18: A[1,1]*B[1,4]
    U[3][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
