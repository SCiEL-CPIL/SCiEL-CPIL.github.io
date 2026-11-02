def construct():
    n, m, p = 2, 3, 8
    # Split m=3 into 1+2
    # Split p=8 into 4+4
    # For M_{2,1,4}: rank 1
    # For M_{2,2,4}: rank 14 (2 Strassen blocks)
    
    # Total rank: 1 + 14 + 14 = 29
    rank = 29
    U = [[0]*rank for _ in range(n*m)]  # 6 x 29
    V = [[0]*rank for _ in range(m*p)]  # 24 x 29
    W = [[0]*rank for _ in range(p*n)]  # 16 x 29
    
    col = 0
    # M_{2,1,4} part: i in {0,1}, j=0, k in {0,1,2,3}
    # Rank 1: U[i*3+0, col] = 1, V[0*8+k, col] = 1, W[k*2+i, col] = 1
    # But this is rank 1, so only one column
    for i in range(2):
        U[i*3+0][col] = 1
    for k in range(4):
        V[0*8+k][col] = 1
    for i in range(2):
        for k in range(4):
            W[k*2+i][col] = 1
    col += 1
    
    # M_{2,2,4} part (first half): i in {0,1}, j in {1,2}, k in {0,1,2,3}
    # Split k into {0,1} and {2,3}, each is M_{2,2,2} with Strassen rank 7
    # Strassen for M_{2,2,2}:
    # m1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
    # m2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
    # m3 = A[i0,j0](B[j0,k1]-B[j1,k1])
    # m4 = A[i1,j1](B[j1,k0]-B[j0,k0])
    # m5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
    # m6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
    # m7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
    # C[i0,k0] = m1+m4-m5+m6
    # C[i0,k1] = m3+m5+m7
    # C[i1,k0] = m2+m4+m6
    # C[i1,k1] = m1-m2+m3+m7
    
    # For j in {1,2}, k in {0,1}:
    j_offset = 1  # j starts at 1
    for k_block in range(2):  # k_block 0: k in {0,1}, k_block 1: k in {2,3}
        k_offset = k_block * 2
        for s in range(7):  # 7 Strassen products
            if s == 0:
                # m1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
                U[0*3+j_offset][col] = 1
                U[1*3+(j_offset+1)][col] = 1
                V[j_offset*8+k_offset][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = 1
                W[k_offset*2+0][col] = 1
                W[(k_offset+1)*2+0][col] = 1
                W[k_offset*2+1][col] = 1
                W[(k_offset+1)*2+1][col] = 1
            elif s == 1:
                # m2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
                U[1*3+j_offset][col] = 1
                U[1*3+(j_offset+1)][col] = 1
                V[j_offset*8+k_offset][col] = 1
                W[k_offset*2+1][col] = 1
            elif s == 2:
                # m3 = A[i0,j0](B[j0,k1]-B[j1,k1])
                U[0*3+j_offset][col] = 1
                V[j_offset*8+(k_offset+1)][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = -1
                W[(k_offset+1)*2+0][col] = 1
            elif s == 3:
                # m4 = A[i1,j1](B[j1,k0]-B[j0,k0])
                U[1*3+(j_offset+1)][col] = 1
                V[(j_offset+1)*8+k_offset][col] = 1
                V[j_offset*8+k_offset][col] = -1
                W[k_offset*2+0][col] = 1
                W[k_offset*2+1][col] = 1
            elif s == 4:
                # m5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
                U[0*3+j_offset][col] = 1
                U[0*3+(j_offset+1)][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = 1
                W[k_offset*2+0][col] = -1
                W[(k_offset+1)*2+0][col] = -1
                W[(k_offset+1)*2+1][col] = 1
            elif s == 5:
                # m6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
                U[1*3+j_offset][col] = 1
                U[0*3+j_offset][col] = -1
                V[j_offset*8+k_offset][col] = 1
                V[j_offset*8+(k_offset+1)][col] = 1
                W[k_offset*2+0][col] = 1
                W[k_offset*2+1][col] = 1
            elif s == 6:
                # m7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
                U[0*3+(j_offset+1)][col] = 1
                U[1*3+(j_offset+1)][col] = -1
                V[(j_offset+1)*8+k_offset][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = 1
                W[(k_offset+1)*2+0][col] = 1
                W[(k_offset+1)*2+1][col] = 1
            col += 1
    
    # M_{2,2,4} part (second half): i in {0,1}, j in {1,2}, k in {4,5,6,7}
    # Same structure, but k_offset starts at 4
    for k_block in range(2):  # k_block 0: k in {4,5}, k_block 1: k in {6,7}
        k_offset = 4 + k_block * 2
        for s in range(7):
            if s == 0:
                U[0*3+j_offset][col] = 1
                U[1*3+(j_offset+1)][col] = 1
                V[j_offset*8+k_offset][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = 1
                W[k_offset*2+0][col] = 1
                W[(k_offset+1)*2+0][col] = 1
                W[k_offset*2+1][col] = 1
                W[(k_offset+1)*2+1][col] = 1
            elif s == 1:
                U[1*3+j_offset][col] = 1
                U[1*3+(j_offset+1)][col] = 1
                V[j_offset*8+k_offset][col] = 1
                W[k_offset*2+1][col] = 1
            elif s == 2:
                U[0*3+j_offset][col] = 1
                V[j_offset*8+(k_offset+1)][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = -1
                W[(k_offset+1)*2+0][col] = 1
            elif s == 3:
                U[1*3+(j_offset+1)][col] = 1
                V[(j_offset+1)*8+k_offset][col] = 1
                V[j_offset*8+k_offset][col] = -1
                W[k_offset*2+0][col] = 1
                W[k_offset*2+1][col] = 1
            elif s == 4:
                U[0*3+j_offset][col] = 1
                U[0*3+(j_offset+1)][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = 1
                W[k_offset*2+0][col] = -1
                W[(k_offset+1)*2+0][col] = -1
                W[(k_offset+1)*2+1][col] = 1
            elif s == 5:
                U[1*3+j_offset][col] = 1
                U[0*3+j_offset][col] = -1
                V[j_offset*8+k_offset][col] = 1
                V[j_offset*8+(k_offset+1)][col] = 1
                W[k_offset*2+0][col] = 1
                W[k_offset*2+1][col] = 1
            elif s == 6:
                U[0*3+(j_offset+1)][col] = 1
                U[1*3+(j_offset+1)][col] = -1
                V[(j_offset+1)*8+k_offset][col] = 1
                V[(j_offset+1)*8+(k_offset+1)][col] = 1
                W[(k_offset+1)*2+0][col] = 1
                W[(k_offset+1)*2+1][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
