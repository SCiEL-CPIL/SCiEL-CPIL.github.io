import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6
    
    # We use the Kronecker product decomposition:
    # 4x4x6 = (2x2x2) (Kronecker) (2x2x3)
    # Rank = Rank(2x2x2) * Rank(2x2x3) = 7 * 11 = 77
    
    # --- Step 1: Define the 2x2x2 Strassen decomposition (Rank 7) ---
    # A (2x2), B (2x2), C (2x2)
    # U1 (4x7), V1 (4x7), W1 (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22) -> c11, c22
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11 -> c21, c22
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22) -> c12, c22
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11) -> c11, c21
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22 -> c11, c12
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> c22
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> c11
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # --- Step 2: Define the 2x2x3 decomposition (Rank 11) ---
    # A (2x2), B (2x3), C (2x3)
    # U2 (4x11), V2 (6x11), W2 (6x11)
    # Indices: U2[i*2+j], V2[j*3+k], W2[k*2+i]
    
    U2 = [[0]*11 for _ in range(4)]
    V2 = [[0]*11 for _ in range(6)]
    W2 = [[0]*11 for _ in range(6)]
    
    # Part 1: Strassen for the first 2 columns of B (cols 0,1) -> C (cols 0,1)
    # This is exactly the 2x2x2 Strassen, but embedded in 2x2x3.
    for c in range(7):
        # U is same as U1
        for i in range(4):
            U2[i][c] = U1[i][c]
        
        # V maps 2x2 B (cols 0,1) to 2x3 B (cols 0,1)
        # V1 row j*2+k -> V2 row j*3+k
        for j in range(2):
            for k in range(2):
                V2[j*3+k][c] = V1[j*2+k][c]
        # Col 2 of B is 0 in this part
        V2[0*3+2][c] = 0
        V2[1*3+2][c] = 0
        
        # W maps 2x2 C (cols 0,1) to 2x3 C (cols 0,1)
        for k in range(2):
            for i in range(2):
                W2[k*2+i][c] = W1[k*2+i][c]
        # Col 2 of C is 0 in this part
        W2[2*2+0][c] = 0
        W2[2*2+1][c] = 0

    # Part 2: Trivial multiplication for the 3rd column of B (col 2) -> C (col 2)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0] * B[0,2] -> C[0,2]
    U2[0][7] = 1
    V2[0*3+2][7] = 1
    W2[2*2+0][7] = 1
    
    # Term 8: A[0,1] * B[1,2] -> C[0,2]
    U2[1][8] = 1
    V2[1*3+2][8] = 1
    W2[2*2+0][8] = 1
    
    # Term 9: A[1,0] * B[0,2] -> C[1,2]
    U2[2][9] = 1
    V2[0*3+2][9] = 1
    W2[2*2+1][9] = 1
    
    # Term 10: A[1,1] * B[1,2] -> C[1,2]
    U2[3][10] = 1
    V2[1*3+2][10] = 1
    W2[2*2+1][10] = 1

    # --- Step 3: Kronecker Product Construction ---
    # Full dimensions: n=4, m=4, p=6
    # Rank = 7 * 11 = 77
    rank = 77
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Index Mapping:
    # A (4x4) = A1 (2x2) (Kronecker) A2 (2x2)
    # i = i1*2 + i2, j = j1*2 + j2
    # U_full[i*4+j, c1*11+c2] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    # B (4x6) = B1 (2x2) (Kronecker) B2 (2x3)
    # j = j1*2 + j2, k = k1*3 + k2
    # V_full[j*6+k, c1*11+c2] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    
    # C (4x6) = C1 (2x2) (Kronecker) C2 (2x3)
    # i = i1*2 + i2, k = k1*3 + k2
    # W_full[k*4+i, c1*11+c2] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            # Fill U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            # Fill V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            # Fill W
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2

    # --- Step 4: Verification ---
    if time.time() < deadline - 5:
        Um = np.array(U)
        Vm = np.array(V)
        Wm = np.array(W)
        
        # Check tensor reconstruction
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We check if T matches the multiplication tensor (1s where indices match)
        
        # Efficient check:
        # For each r, we have a rank-1 term.
        # We can check a few random points or all points if fast enough.
        # 16*24*24 = 9216 points. 77 terms. 77 * 9216 ~ 700k ops. Very fast.
        
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            # Vectorized addition
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0: continue
                    for k in range(p):
                        T_recon[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1.0
        
        if not np.allclose(T_recon, T_target):
            # Fallback to schoolbook if something went wrong (should not happen)
            rank2 = n * m * p
            U2 = [[0] * rank2 for _ in range(n * m)]
            V2 = [[0] * rank2 for _ in range(m * p)]
            W2 = [[0] * rank2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i*m+j][r] = 1
                        V2[j*p+k][r] = 1
                        W2[k*n+i][r] = 1
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
