import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:, t], V[:, t])
            T_recon += np.einsum('ab,c->abc', uv, W[:, t])
        return np.max(np.abs(T_recon - T))

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)

    # Strassen-3: rank-7 decomposition for 3x3 * 3x3
    # Returns U (9,7), V (9,7), W (9,7)
    def strassen_3x3():
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # U indices: i*3+j (row i, col j of X)
        # V indices: j*3+k (row j, col k of Y)
        # W indices: k*3+i (col k, row i of C)
        
        # m1 = x[0,0] * (y[1,1] - y[1,2] - y[2,1] + y[2,2])
        t = 0
        U[0, t] = 1
        V[1*3+1, t] = 1   # y[1,1]
        V[1*3+2, t] = -1  # y[1,2]
        V[2*3+1, t] = -1  # y[2,1]
        V[2*3+2, t] = 1   # y[2,2]
        W[0*3+0, t] = 1   # c[0,0] (k=0,i=0)
        W[0*3+1, t] = 1   # c[1,0] (k=0,i=1)
        W[0*3+2, t] = 1   # c[2,0] (k=0,i=2)
        
        # m2 = (x[0,0] + x[0,1]) * (y[1,2] + y[2,2] - y[1,1] - y[2,1])
        t = 1
        U[0, t] = 1
        U[1, t] = 1
        V[1*3+2, t] = 1   # y[1,2]
        V[2*3+2, t] = 1   # y[2,2]
        V[1*3+1, t] = -1  # y[1,1]
        V[2*3+1, t] = -1  # y[2,1]
        W[1*3+0, t] = 1   # c[0,1] (k=1,i=0)
        W[1*3+1, t] = 1   # c[1,1] (k=1,i=1)
        W[1*3+2, t] = 1   # c[2,1] (k=1,i=2)
        
        # m3 = (x[0,1] + x[0,2]) * (y[1,0] + y[2,0] - y[1,1] - y[2,1])
        t = 2
        U[1, t] = 1
        U[2, t] = 1
        V[1*3+0, t] = 1   # y[1,0]
        V[2*3+0, t] = 1   # y[2,0]
        V[1*3+1, t] = -1  # y[1,1]
        V[2*3+1, t] = -1  # y[2,1]
        W[2*3+0, t] = 1   # c[0,2] (k=2,i=0)
        W[2*3+1, t] = 1   # c[1,2] (k=2,i=1)
        W[2*3+2, t] = 1   # c[2,2] (k=2,i=2)
        
        # m4 = x[2,2] * (y[0,0] - y[0,1] - y[1,0] + y[1,1])
        t = 3
        U[8, t] = 1
        V[0*3+0, t] = 1   # y[0,0]
        V[0*3+1, t] = -1  # y[0,1]
        V[1*3+0, t] = -1  # y[1,0]
        V[1*3+1, t] = 1   # y[1,1]
        W[0*3+0, t] = 1   # c[0,0]
        W[1*3+0, t] = 1   # c[0,1]
        W[2*3+0, t] = 1   # c[0,2]
        
        # m5 = (x[2,0] + x[2,2]) * (y[0,1] + y[1,1] - y[0,0] - y[1,0])
        t = 4
        U[6, t] = 1
        U[8, t] = 1
        V[0*3+1, t] = 1   # y[0,1]
        V[1*3+1, t] = 1   # y[1,1]
        V[0*3+0, t] = -1  # y[0,0]
        V[1*3+0, t] = -1  # y[1,0]
        W[0*3+1, t] = 1   # c[1,0]
        W[1*3+1, t] = 1   # c[1,1]
        W[2*3+1, t] = 1   # c[2,1]
        
        # m6 = (x[2,1] + x[2,2]) * (y[0,0] - y[0,1])
        t = 5
        U[7, t] = 1
        U[8, t] = 1
        V[0*3+0, t] = 1   # y[0,0]
        V[0*3+1, t] = -1  # y[0,1]
        W[0*3+2, t] = 1   # c[2,0]
        W[1*3+2, t] = 1   # c[2,1]
        W[2*3+2, t] = 1   # c[2,2]
        
        # m7 = (x[0,0] + x[1,1] + x[2,2]) * (y[0,0] + y[1,1] + y[2,2])
        t = 6
        U[0, t] = 1
        U[4, t] = 1
        U[8, t] = 1
        V[0*3+0, t] = 1   # y[0,0]
        V[1*3+1, t] = 1   # y[1,1]
        V[2*3+2, t] = 1   # y[2,2]
        W[0*3+0, t] = 1   # c[0,0]
        W[1*3+0, t] = 1   # c[0,1]
        W[2*3+0, t] = 1   # c[0,2]
        W[0*3+1, t] = 1   # c[1,0]
        W[1*3+1, t] = 1   # c[1,1]
        W[2*3+1, t] = 1   # c[2,1]
        W[0*3+2, t] = 1   # c[2,0]
        W[1*3+2, t] = 1   # c[2,1]
        W[2*3+2, t] = 1   # c[2,2]
        
        return U, V, W
    
    # Verify Strassen-3
    U_s3, V_s3, W_s3 = strassen_3x3()
    T_33 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_33[i*3+j, j*3+k, k*3+i] = 1.0
    
    T_recon_33 = np.zeros((9, 9, 9))
    for t in range(7):
        uv = np.outer(U_s3[:, t], V_s3[:, t])
        T_recon_33 += np.einsum('ab,c->abc', uv, W_s3[:, t])
    
    err_33 = np.max(np.abs(T_recon_33 - T_33))
    
    if err_33 > 1e-10:
        # Strassen-3 construction has errors, fall back to schoolbook
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the full 4x5*5x6 decomposition using Strassen-3 on 3x3x3 blocks
    #
    # Partition:
    # n=4: I = {0,1,2} (3 rows), I' = {3} (1 row)
    # m=5: J = {0,1,2} (3 cols), J' = {3,4} (2 cols)
    # p=6: K = {0,1,2} (3 cols), K' = {3,4,5} (3 cols)
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We need all combinations of (I/I') x (J/J') x (J/J') x (K/K'):
    # The sum over j splits into j in J and j in J'.
    #
    # Block (I, J, K): A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    #   B is 3x3, can use Strassen-3: rank 7
    # Block (I, J, K'): A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]
    #   B is 3x3, can use Strassen-3: rank 7
    # Block (I, J', K): A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3]
    #   A is 3x2, B is 2x3: schoolbook 3*2*3=18
    # Block (I, J', K'): A[0:3,3:5] * B[3:5,3:6] -> C[0:3,3:6]
    #   A is 3x2, B is 2x3: schoolbook 3*2*3=18
    # Block (I', J, K): A[3,0:3] * B[0:3,0:3] -> C[3,0:3]
    #   A is 1x3, B is 3x3: schoolbook 1*3*3=9
    # Block (I', J, K'): A[3,0:3] * B[0:3,3:6] -> C[3,3:6]
    #   A is 1x3, B is 3x3: schoolbook 1*3*3=9
    # Block (I', J', K): A[3,3:5] * B[3:5,0:3] -> C[3,0:3]
    #   A is 1x2, B is 2x3: schoolbook 1*2*3=6
    # Block (I', J', K'): A[3,3:5] * B[3:5,3:6] -> C[3,3:6]
    #   A is 1x2, B is 2x3: schoolbook 1*2*3=6
    #
    # Total: 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6 = 80
    
    r_total = 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6
    U_full = np.zeros((nm, r_total))
    V_full = np.zeros((mp, r_total))
    W_full = np.zeros((pn, r_total))
    
    col = 0
    
    # Block (I,J,K): A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Strassen-3
    for t in range(7):
        for a_idx in range(9):
            if U_s3[a_idx, t] != 0:
                i = a_idx // 3
                j = a_idx % 3
                U_full[i*5 + j, col] += U_s3[a_idx, t]
        for b_idx in range(9):
            if V_s3[b_idx, t] != 0:
                j = b_idx // 3
                k = b_idx % 3
                V_full[j*6 + k, col] += V_s3[b_idx, t]
        for c_idx in range(9):
            if W_s3[c_idx, t] != 0:
                k = c_idx // 3
                i = c_idx % 3
                W_full[k*4 + i, col] += W_s3[c_idx, t]
        col += 1
    
    # Block (I,J,K'): A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    # Strassen-3 (B block is 3x3: rows 0-2 of B, cols 3-5)
    for t in range(7):
        for a_idx in range(9):
            if U_s3[a_idx, t] != 0:
                i = a_idx // 3
                j = a_idx % 3
                U_full[i*5 + j, col] += U_s3[a_idx, t]
        for b_idx in range(9):
            if V_s3[b_idx, t] != 0:
                j = b_idx // 3
                k = b_idx % 3
                V_full[j*6 + (3 + k), col] += V_s3[b_idx, t]
        for c_idx in range(9):
            if W_s3[c_idx, t] != 0:
                k = c_idx // 3
                i = c_idx % 3
                W_full[(3 + k)*4 + i, col] += W_s3[c_idx, t]
        col += 1
    
    # Block (I,J',K): A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]
    # Schoolbook: 3*2*3 = 18
    for i in range(3):
        for jj in range(2):
            j = 3 + jj
            for k in range(3):
                U_full[i*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + i, col] = 1
                col += 1
    
    # Block (I,J',K'): A[0:3, 3:5] * B[3:5, 3:6] -> C[0:3, 3:6]
    # Schoolbook: 3*2*3 = 18
    for i in range(3):
        for jj in range(2):
            j = 3 + jj
            for kk in range(3):
                k = 3 + kk
                U_full[i*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + i, col] = 1
                col += 1
    
    # Block (I',J,K): A[3, 0:3] * B[0:3, 0:3] -> C[3, 0:3]
    # Schoolbook: 1*3*3 = 9
    for j in range(3):
        for k in range(3):
            U_full[3*5 + j, col] = 1
            V_full[j*6 + k, col] = 1
            W_full[k*4 + 3, col] = 1
            col += 1
    
    # Block (I',J,K'): A[3, 0:3] * B[0:3, 3:6] -> C[3, 3:6]
    # Schoolbook: 1*3*3 = 9
    for j in range(3):
        for kk in range(3):
            k = 3 + kk
            U_full[3*5 + j, col] = 1
            V_full[j*6 + k, col] = 1
            W_full[k*4 + 3, col] = 1
            col += 1
    
    # Block (I',J',K): A[3, 3:5] * B[3:5, 0:3] -> C[3, 0:3]
    # Schoolbook: 1*2*3 = 6
    for jj in range(2):
        j = 3 + jj
        for k in range(3):
            U_full[3*5 + j, col] = 1
            V_full[j*6 + k, col] = 1
            W_full[k*4 + 3, col] = 1
            col += 1
    
    # Block (I',J',K'): A[3, 3:5] * B[3:5, 3:6] -> C[3, 3:6]
    # Schoolbook: 1*2*3 = 6
    for jj in range(2):
        j = 3 + jj
        for kk in range(3):
            k = 3 + kk
            U_full[3*5 + j, col] = 1
            V_full[j*6 + k, col] = 1
            W_full[k*4 + 3, col] = 1
            col += 1
    
    # Verify
    err_full = verify_decomposition(U_full, V_full, W_full)
    
    if err_full < 1e-10:
        best_rank = r_total
        best_decomp = (U_full.copy(), V_full.copy(), W_full.copy())
    else:
        # Fall back to schoolbook
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
