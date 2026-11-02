"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Block structure: split n=4 as 2+2, m=4 as 2+2, p=6 as 3+3
    # A[i,j] with i=i1*2+i2, j=j1*2+j2
    # B[j,k] with j=j1*2+j2, k=k1*3+k2  
    # C[i,k] with i=i1*2+i2, k=k1*3+k2
    # W[k*n+i] convention
    
    # C[i1,i2, k1,k2] = sum_{j1,j2} A[i1,i2,j1,j2] * B[j1,j2,k1,k2]
    # = A[i1,i2,0,0]*B[0,0,k1,k2] + A[i1,i2,0,1]*B[0,1,k1,k2]
    #   + A[i1,i2,1,0]*B[1,0,k1,k2] + A[i1,i2,1,1]*B[1,1,k1,k2]
    
    # For fixed (i1,i2,k1,k2), this is a 2x2x(2x3) = 2x2x3 type product
    # But more precisely, for each (i1,i2) pair, we compute a 2x3 matrix C_block
    # C_block = A_block(2x2) * B_blocks(2x2 and 2x3 interleaved)
    
    # Let's use: for each (i1, k1) in {0,1}x{0,1}, and each i2 in {0,1}:
    # C[i1,i2, k1, :] = A[i1,i2, 0, :] * B[0, :, k1, :] + A[i1,i2, 1, :] * B[1, :, k1, :]
    
    # This is 2x2x3 matrix multiplication for each (i1, i2, k1) triple
    # 2*2*2 = 8 such subproblems, each of type 2x2x3
    
    # Rank of 2x2x3: use schoolbook rank 12 for simplicity, total = 8*12 = 96
    # Or use rank 6 if we can find it, total = 8*6 = 48
    
    # Let's use a rank-8 decomposition for 2x2x3 (better than 12, simpler than 6)
    # Actually, let me just use schoolbook for correctness: rank 12 per subproblem
    # Total rank = 8 * 12 = 96. Worse than 84.
    
    # Better: use the Kronecker product approach from inspiration 1
    # 4x4x6 = (2x2x2) kron (2x2x3)
    # Rank = 7 * 12 = 84
    
    # Strassen for 2x2x2
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a10+a11)b00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a00(b10-b11)
    U1[0][2] = 1
    V1[2][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a11(b01-b00)
    U1[3][3] = 1
    V1[1][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a00+a01)b11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # Schoolbook for 2x2x3
    U2 = [[0]*12 for _ in range(4)]
    V2 = [[0]*12 for _ in range(6)]
    W2 = [[0]*12 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U2[i*2+j][col] = 1
                V2[j*3+k][col] = 1
                W2[k*2+i][col] = 1
                col += 1
    
    # Kronecker product
    rank = 7 * 12
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(7):
        for c2 in range(12):
            c = c1 * 12 + c2
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
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    if not valid:
        # Fallback to schoolbook
        rank_s = n * m * p
        U_s = [[0]*rank_s for _ in range(n*m)]
        V_s = [[0]*rank_s for _ in range(m*p)]
        W_s = [[0]*rank_s for _ in range(p*n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col_s] = 1
                    V_s[j*p+k][col_s] = 1
                    W_s[k*n+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
