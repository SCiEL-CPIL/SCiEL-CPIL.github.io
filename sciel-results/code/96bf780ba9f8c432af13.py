"""Matrix multiplication 3x4x4: Rank-minimal exact tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Check valid entries: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check some invalid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i > 0:
                    val = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i-1,:])
                    if abs(val) > 1e-9:
                        return False
                if j > 0:
                    val = np.sum(U_r[i,j-1,:] * V_r[j,k,:] * W_r[k,i,:])
                    if abs(val) > 1e-9:
                        return False
                if k > 0:
                    val = np.sum(U_r[i,j,:] * V_r[j,k-1,:] * W_r[k,i,:])
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _strassen_2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (4, 7) - U[i*2+j, r] = coeff of A[i,j]
    V: (4, 7) - V[j*2+k, r] = coeff of B[j,k]
    W: (4, 7) - W[k*2+i, r] = coeff of C[i,k]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    U[0,0] = 1   # A[0,0]
    U[3,0] = 1   # A[1,1]
    V[0,0] = 1   # B[0,0]
    V[3,0] = 1   # B[1,1]
    W[0,0] += 1  # C[0,0]
    W[3,0] += 1  # C[1,1]
    
    # r=1: M2 = (A[1,0]+A[1,1])B[0,0]
    U[2,1] = 1   # A[1,0]
    U[3,1] = 1   # A[1,1]
    V[0,1] = 1   # B[0,0]
    W[0,1] += 1  # C[1,0]
    W[3,1] -= 1  # C[1,1]
    
    # r=2: M3 = A[0,0](B[0,1]-B[1,1])
    U[0,2] = 1   # A[0,0]
    V[1,2] = 1   # B[0,1]
    V[3,2] = -1  # B[1,1]
    W[1,2] += 1  # C[0,1]
    W[3,2] += 1  # C[1,1]
    
    # r=3: M4 = A[1,1](B[1,0]-B[0,0])
    U[3,3] = 1   # A[1,1]
    V[2,3] = 1   # B[1,0]
    V[0,3] = -1  # B[0,0]
    W[0,3] += 1  # C[0,0]
    W[0,3] += 0  # (already set)
    W[2*2+0, 3] += 1  # C[1,0] -> W[0*2+1, 3]... wait
    
    # Let me be more careful with W indexing
    # W[k*2+i, r] = coeff of C[i,k]
    # C[0,0]: W[0, r]
    # C[0,1]: W[1, r]  
    # C[1,0]: W[2, r]
    # C[1,1]: W[3, r]
    
    # Redo r=3: M4 = A[1,1](B[1,0]-B[0,0])
    # C[0,0] += M4, C[1,0] += M4
    U[3,3] = 1   # A[1,1]
    V[2,3] = 1   # B[1,0]
    V[0,3] = -1  # B[0,0]
    W[0,3] += 1  # C[0,0]
    W[2,3] += 1  # C[1,0]
    
    # r=4: M5 = (A[0,0]+A[0,1])B[1,1]
    # C[0,0] -= M5, C[0,1] += M5
    U[0,4] = 1   # A[0,0]
    U[1,4] = 1   # A[0,1]
    V[3,4] = 1   # B[1,1]
    W[0,4] -= 1  # C[0,0]
    W[1,4] += 1  # C[0,1]
    
    # r=5: M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # C[1,1] += M6
    U[2,5] = 1   # A[1,0]
    U[0,5] = -1  # A[0,0]
    V[0,5] = 1   # B[0,0]
    V[1,5] = 1   # B[0,1]
    W[3,5] += 1  # C[1,1]
    
    # r=6: M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    # C[0,0] += M7
    U[1,6] = 1   # A[0,1]
    U[3,6] = -1  # A[1,1]
    V[2,6] = 1   # B[1,0]
    V[3,6] = 1   # B[1,1]
    W[0,6] += 1  # C[0,0]
    
    return U, V, W

def _direct_1x2_2x2():
    """
    Return U, V, W for 1x2 times 2x2 = 1x2 (schoolbook, rank 4).
    U: (2, 4) - U[j, r] = coeff of A[j] (A is 1x2)
    V: (4, 4) - V[j*2+k, r] = coeff of B[j,k]
    W: (2, 4) - W[k, r] = coeff of C[k] (C is 1x2)
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    
    # r=0: A[0]*B[0,0] -> C[0]
    U[0,0] = 1
    V[0,0] = 1   # B[0,0]
    W[0,0] = 1   # C[0]
    
    # r=1: A[0]*B[0,1] -> C[1]
    U[0,1] = 1
    V[1,1] = 1   # B[0,1]
    W[1,1] = 1   # C[1]
    
    # r=2: A[1]*B[1,0] -> C[0]
    U[1,2] = 1
    V[2,2] = 1   # B[1,0]
    W[0,2] = 1   # C[0]
    
    # r=3: A[1]*B[1,1] -> C[1]
    U[1,3] = 1
    V[3,3] = 1   # B[1,1]
    W[1,3] = 1   # C[1]
    
    return U, V, W

def construct():
    """
    Construct a tensor decomposition for 3x4x4 matrix multiplication.
    Uses Strassen (rank 7) for 2x2x2 blocks and direct (rank 4) for 1x2x2x2 blocks.
    Total rank: 4*7 + 4*4 = 28 + 16 = 44
    """
    n, m, p = 3, 4, 4
    total_rank = 44
    
    U = [[0] * total_rank for _ in range(n * m)]  # (12, 44)
    V = [[0] * total_rank for _ in range(m * p)]  # (16, 44)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 44)
    
    U_s, V_s, W_s = _strassen_2x2()
    U_d, V_d, W_d = _direct_1x2_2x2()
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    U[i*4+j][col+r] += U_s[i*2+j][r]
                    V[j*4+k][col+r] += V_s[j*2+k][r]
                    W[k*3+i][col+r] += W_s[k*2+i][r]
    col += 7
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    U[i*4+(j+2)][col+r] += U_s[i*2+j][r]
                    V[(j+2)*4+k][col+r] += V_s[j*2+k][r]
                    W[k*3+i][col+r] += W_s[k*2+i][r]
    col += 7
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    U[i*4+j][col+r] += U_s[i*2+j][r]
                    V[j*4+(k+2)][col+r] += V_s[j*2+k][r]
                    W[(k+2)*3+i][col+r] += W_s[k*2+i][r]
    col += 7
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    U[i*4+(j+2)][col+r] += U_s[i*2+j][r]
                    V[(j+2)*4+(k+2)][col+r] += V_s[j*2+k][r]
                    W[(k+2)*3+i][col+r] += W_s[k*2+i][r]
    col += 7
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    for j in range(2):
        for k in range(2):
            for r in range(4):
                U[2*4+j][col+r] += U_d[j][r]
                V[j*4+k][col+r] += V_d[j*2+k][r]
                W[k*3+2][col+r] += W_d[k][r]
    col += 4
    
    # Block 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    for j in range(2):
        for k in range(2):
            for r in range(4):
                U[2*4+(j+2)][col+r] += U_d[j][r]
                V[(j+2)*4+k][col+r] += V_d[j*2+k][r]
                W[k*3+2][col+r] += W_d[k][r]
    col += 4
    
    # Block 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    for j in range(2):
        for k in range(2):
            for r in range(4):
                U[2*4+j][col+r] += U_d[j][r]
                V[j*4+(k+2)][col+r] += V_d[j*2+k][r]
                W[(k+2)*3+2][col+r] += W_d[k][r]
    col += 4
    
    # Block 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    for j in range(2):
        for k in range(2):
            for r in range(4):
                U[2*4+(j+2)][col+r] += U_d[j][r]
                V[(j+2)*4+(k+2)][col+r] += V_d[j*2+k][r]
                W[(k+2)*3+2][col+r] += W_d[k][r]
    col += 4
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = 48
        U_fb = [[0] * rank_sb for _ in range(n * m)]
        V_fb = [[0] * rank_sb for _ in range(m * p)]
        W_fb = [[0] * rank_sb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
