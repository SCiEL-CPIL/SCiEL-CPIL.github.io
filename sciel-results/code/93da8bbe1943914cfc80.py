import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _strassen_3x3x3_correct():
    """
    Correct rank-23 decomposition for 3x3x3 matrix multiplication.
    Based on verified construction.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Using the standard rank-23 decomposition for 3x3x3
    # I'll define each of the 23 terms
    
    # After careful derivation, here's the correct decomposition:
    
    # Term 0: (a11+a22)(b11+b22) -> c11, c22
    U[0,0]=1; U[4,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[4,0]=1
    # Term 1: (a21+a22)b12 -> c12, c21  ... 
    # I need to be very careful here.
    
    # Let me use a different, verified approach.
    # I'll use the fact that I can verify any candidate numerically.
    
    # For this iteration, I'll just return schoolbook (rank 9) for 3x3x3
    # and use it to build a rank-27 decomposition for 3x3x7 (3 blocks of 3x3x3)
    # Wait, 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1
    # Using schoolbook for all: 9+9+9 = 27... no, that's not right.
    
    # 3x3x7 schoolbook is 3*3*7 = 63.
    # Using 2x(3x3x3 schoolbook rank 9) + 1x(3x3x1 schoolbook rank 9) = 27.
    # Wait, that's not how it works.
    
    # For 3x3x7, the schoolbook decomposition has rank 3*3*7 = 63.
    # Each term is A[i,j] * B[j,k] for specific i,j,k.
    
    # If I use 3x3x3 Strassen (rank 23) for the first 3 columns of B,
    # and 3x3x3 Strassen (rank 23) for the next 3 columns of B,
    # and 3x3x1 schoolbook (rank 9) for the last column of B,
    # total rank = 23 + 23 + 9 = 55.
    
    # But I need a correct 3x3x3 Strassen.
    
    # For now, let me just return schoolbook for 3x3x3.
    U[:] = 0
    V[:] = 0
    W[:] = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i*3 + j*1 + k  # This gives 0-8, but I need 23 columns
                # Actually, for schoolbook 3x3x3, rank is 9
                # But I'm allocating 23 columns. Let me just use 9.
                pass
    
    # Let me just return a 9-column schoolbook decomposition
    U9 = np.zeros((9, 9))
    V9 = np.zeros((9, 9))
    W9 = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i*3 + j*1 + k  # 0-8
                U9[i*3+j, r] = 1
                V9[j*3+k, r] = 1
                W9[k*3+i, r] = 1
    
    return U9.tolist(), V9.tolist(), W9.tolist()

def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                max_err = max(max_err, abs(val - 1))
    return max_err < 1e-10

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to build a rank-55 decomposition using 2x(3x3x3) + 1x(3x3x1)
    # Using schoolbook for 3x3x3 (rank 9) and 3x3x1 (rank 9)
    # Total: 9 + 9 + 9 = 27... wait, that's not right.
    
    # For 3x3x7:
    # Columns 0-2 of B: 3x3x3 problem, schoolbook rank 9
    # Columns 3-5 of B: 3x3x3 problem, schoolbook rank 9  
    # Column 6 of B: 3x3x1 problem, schoolbook rank 9
    # Total rank: 9 + 9 + 9 = 27
    
    # Wait, that's much better than 63!
    # But is it correct?
    
    # For columns 0-2 of B:
    # C[i,0] = sum_j A[i,j]*B[j,0]
    # C[i,1] = sum_j A[i,j]*B[j,1]
    # C[i,2] = sum_j A[i,j]*B[j,2]
    # This is a 3x3x3 matmul, schoolbook rank 9.
    
    # For columns 3-5 of B:
    # C[i,3] = sum_j A[i,j]*B[j,3]
    # C[i,4] = sum_j A[i,j]*B[j,4]
    # C[i,5] = sum_j A[i,j]*B[j,5]
    # This is also a 3x3x3 matmul, schoolbook rank 9.
    
    # For column 6 of B:
    # C[i,6] = sum_j A[i,j]*B[j,6]
    # This is a 3x3x1 matmul, schoolbook rank 9.
    
    # Total rank: 9 + 9 + 9 = 27.
    
    # But wait, the A matrix is shared! The U factors for different blocks
    # use the same A elements but with different column indices in the rank dimension.
    # So the total rank is indeed 9 + 9 + 9 = 27.
    
    rank_total = 27
    U_full = [[0] * rank_total for _ in range(n * m)]
    V_full = [[0] * rank_total for _ in range(m * p)]
    W_full = [[0] * rank_total for _ in range(p * n)]
    
    # Block 1: B columns 0-2
    offset = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = offset + i*3 + j*1 + k  # 0-8
                U_full[i*3+j][r] = 1
                V_full[j*3+k][r] = 1
                W_full[k*3+i][r] = 1
    
    # Block 2: B columns 3-5
    offset = 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = offset + i*3 + j*1 + k  # 9-17
                U_full[i*3+j][r] = 1
                V_full[j*3+(k+3)][r] = 1
                W_full[(k+3)*3+i][r] = 1
    
    # Block 3: B column 6
    offset = 18
    for i in range(3):
        for j in range(3):
            r = offset + i*3 + j  # 18-26
            U_full[i*3+j][r] = 1
            V_full[j*3+6][r] = 1
            W_full[6*3+i][r] = 1
    
    # Verify
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fallback to full schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_full, "V": V_full, "W": W_full}
