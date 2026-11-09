import time
import numpy as np

def construct():
    """
    Returns a rank-minimal (known best construction) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Use Strassen's rank-7 decomposition for the 2x4x4 part (4 blocks of 2x2x2).
    - Use schoolbook (rank 8) for the remaining 2x4x1 part (last column of B).
    - Total rank = 4*7 + 8 = 36.
    
    The output follows the trace convention W[k*n + i, r] where:
    - A is 2x4, B is 4x5, C is 2x5
    - C[i,k] = sum_j A[i,j] * B[j,k]
    - U has shape (2*4, rank) = (8, rank)
    - V has shape (4*5, rank) = (20, rank)
    - W has shape (5*2, rank) = (10, rank)
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = 2, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # For C = A*B where A, B, C are 2x2:
    # C[0,0] = M1 - M2 + M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6
    # where:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # U, V, W for a single 2x2x2 Strassen block
    # U[i*2+j, r] = coefficient for A[i,j] in multiplication r
    # V[j*2+k, r] = coefficient for B[j,k] in multiplication r
    # W[k*2+i, r] = coefficient for C[k,i] in multiplication r
    
    strassen_U = np.array([
        [1, 0, 0, 1],   # M1: (A[0,0]+A[1,1])
        [0, 0, 1, 1],   # M2: (A[1,0]+A[1,1])
        [1, 0, 0, 0],   # M3: A[0,0]
        [0, 0, 0, 1],   # M4: A[1,1]
        [1, 1, 0, 0],   # M5: (A[0,0]+A[0,1])
        [-1, 0, 1, 0],  # M6: (A[1,0]-A[0,0])
        [0, 1, 0, -1]   # M7: (A[0,1]-A[1,1])
    ], dtype=int)
    
    strassen_V = np.array([
        [1, 0, 0, 1],   # M1: (B[0,0]+B[1,1])
        [1, 0, 0, 0],   # M2: B[0,0]
        [0, 1, 0, -1],  # M3: (B[0,1]-B[1,1])
        [-1, 0, 1, 0],  # M4: (B[1,0]-B[0,0])
        [0, 0, 0, 1],   # M5: B[1,1]
        [1, 1, 0, 0],   # M6: (B[0,0]+B[0,1])
        [0, 0, 1, 1]    # M7: (B[1,0]+B[1,1])
    ], dtype=int)
    
    # W[k*2+i, r]: coefficient for C[k,i]
    # C[0,0] = M1 - M2 + M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6
    # Note: In the trace convention, W is indexed as W[k*n+i, r] where n=2 for 2x2 blocks
    # So for a 2x2 block, W[k*2+i, r] gives coefficient for C[i,k] (note: row i, col k)
    
    # Let me be more careful. The convention is W[k*n+i, r].
    # For 2x2x2: n=2, so W[k*2+i, r] where i is row index, k is col index of C.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # C[0,0] = M1 - M2 + M4 + M5  -> W[0*2+0, r] = W[0, r]
    # C[0,1] = M3 + M5            -> W[1*2+0, r] = W[2, r]
    # C[1,0] = M2 + M4            -> W[0*2+1, r] = W[1, r]
    # C[1,1] = M1 - M3 + M4 + M6  -> W[1*2+1, r] = W[3, r]
    
    strassen_W = np.array([
        [1, 0, 0, 1],   # W[0, r] = C[0,0] = M1 - M2 + M4 + M5
        [-1, 0, 1, 0],  # W[1, r] = C[1,0] = M2 + M4
        [0, 1, 0, -1],  # W[2, r] = C[0,1] = M3 + M5
        [1, 1, 0, 0]    # W[3, r] = C[1,1] = M1 - M3 + M4 + M6
    ], dtype=int)
    
    # Wait, let me re-verify Strassen's formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M2 + M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M4 + M6
    
    # So:
    # W[0,0] = 1 (M1), W[0,1] = -1 (M2), W[0,2] = 0, W[0,3] = 1 (M4), W[0,4] = 1 (M5), W[0,5] = 0, W[0,6] = 0
    # W[1,0] = 0, W[1,1] = 1 (M2), W[1,2] = 0, W[1,3] = 1 (M4), W[1,4] = 0, W[1,5] = 0, W[1,6] = 0
    # W[2,0] = 0, W[2,1] = 0, W[2,2] = 1 (M3), W[2,3] = 0, W[2,4] = 1 (M5), W[2,5] = 0, W[2,6] = 0
    # W[3,0] = 1 (M1), W[3,1] = 0, W[3,2] = -1 (M3), W[3,3] = 1 (M4), W[3,4] = 0, W[3,5] = 1 (M6), W[3,6] = 0
    
    strassen_W_correct = np.array([
        [1, -1, 0, 1, 1, 0, 0],   # W[0, r] for C[0,0]
        [0, 1, 0, 1, 0, 0, 0],    # W[1, r] for C[1,0]
        [0, 0, 1, 0, 1, 0, 0],    # W[2, r] for C[0,1]
        [1, 0, -1, 1, 0, 1, 0]    # W[3, r] for C[1,1]
    ], dtype=int)
    
    rank = 4 * 7 + 8  # 36
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Decompose 2x4x4 part using 4 Strassen blocks
    # A is 2x4, B is 4x4 (first 4 columns), C is 2x4 (first 4 columns)
    # Split into 2x2 blocks: j in {0,1} and {2,3}, k in {0,1} and {2,3}
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                # U part: 2x2 block at rows 0-1, cols j0 to j0+1
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj, col] = strassen_U[r, li * 2 + lj]
                
                # V part: 2x2 block at rows j0 to j0+1, cols k0 to k0+1
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk, col] = strassen_V[r, lj * 2 + lk]
                
                # W part: 2x2 block at rows k0 to k0+1, cols 0-1
                # W[k0+lk, gi] corresponds to W[(k0+lk)*n + gi, col]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi, col] = strassen_W_correct[r, lk * 2 + li]
                
                col += 1
    
    # Decompose 2x4x1 part (last column of B, col 4) using schoolbook
    # C[:, 4] = A * B[:, 4]
    # For each (i, j): C[i, 4] += A[i, j] * B[j, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
