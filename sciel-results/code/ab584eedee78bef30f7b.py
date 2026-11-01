"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Deterministic rank-44 construction:
    # Decompose M_{2,3,8} into 4 blocks of M_{2,3,2}
    # Each block: A(2x3) * B(3x2) = C(2x2)
    # Per block: Strassen-7 on 2x2x2 core + 4 rank-1 for extra row = rank 11
    # Total rank: 4 * 11 = 44
    
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        # For block blk, B columns are 2*blk and 2*blk+1
        # C columns (output) are 2*blk and 2*blk+1
        # W index for C[i, k] = k*n + i = 2k + i
        # For block blk: k = 2*blk or 2*blk+1
        #   C[i, 2*blk]   -> W[4*blk + i]
        #   C[i, 2*blk+1] -> W[4*blk + 2 + i]
        
        # Strassen part (indices 0-6 in block):
        # m1 = (A00+A11)(B00+B11)
        U[0, base + 0] = 1   # A[0,0]
        U[4, base + 0] = 1   # A[1,1]
        V[2 * blk, base + 0] = 1           # B[0, 2*blk]
        V[p + 2 * blk + 1, base + 0] = 1   # B[1, 2*blk+1]
        W[4 * blk + 0, base + 0] = 1       # C[0, 2*blk]
        W[4 * blk + 3, base + 0] = 1       # C[1, 2*blk+1]
        
        # m2 = (A10+A11)B00
        U[3, base + 1] = 1   # A[1,0]
        U[4, base + 1] = 1   # A[1,1]
        V[2 * blk, base + 1] = 1           # B[0, 2*blk]
        W[4 * blk + 1, base + 1] = 1       # C[1, 2*blk]
        W[4 * blk + 3, base + 1] = -1      # C[1, 2*blk+1]
        
        # m3 = A00(B01-B11)
        U[0, base + 2] = 1   # A[0,0]
        V[2 * blk + 1, base + 2] = 1     # B[0, 2*blk+1]
        V[p + 2 * blk + 1, base + 2] = -1  # B[1, 2*blk+1]
        W[4 * blk + 2, base + 2] = 1     # C[0, 2*blk+1]
        W[4 * blk + 3, base + 2] = 1     # C[1, 2*blk+1]
        
        # m4 = A11(B10-B00)
        U[4, base + 3] = 1   # A[1,1]
        V[p + 2 * blk, base + 3] = 1     # B[1, 2*blk]
        V[2 * blk, base + 3] = -1        # B[0, 2*blk]
        W[4 * blk + 0, base + 3] = 1     # C[0, 2*blk]
        W[4 * blk + 1, base + 3] = 1     # C[1, 2*blk]
        
        # m5 = (A00+A01)B11
        U[0, base + 4] = 1   # A[0,0]
        U[1, base + 4] = 1   # A[0,1]
        V[p + 2 * blk + 1, base + 4] = 1  # B[1, 2*blk+1]
        W[4 * blk + 0, base + 4] = -1     # C[0, 2*blk]
        W[4 * blk + 2, base + 4] = 1      # C[0, 2*blk+1]
        
        # m6 = (A10-A00)(B00+B01)
        U[3, base + 5] = 1   # A[1,0]
        U[0, base + 5] = -1  # A[0,0]
        V[2 * blk, base + 5] = 1          # B[0, 2*blk]
        V[2 * blk + 1, base + 5] = 1      # B[0, 2*blk+1]
        W[4 * blk + 3, base + 5] = 1      # C[1, 2*blk+1]
        
        # m7 = (A01-A11)(B10+B11)
        U[1, base + 6] = 1   # A[0,1]
        U[4, base + 6] = -1  # A[1,1]
        V[p + 2 * blk, base + 6] = 1          # B[1, 2*blk]
        V[p + 2 * blk + 1, base + 6] = 1      # B[1, 2*blk+1]
        W[4 * blk + 0, base + 6] = 1          # C[0, 2*blk]
        
        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # 4 rank-1 terms (indices 7-10 in block):
        # C[0, 2*blk] += A[0,2]*B[2, 2*blk]
        U[2, base + 7] = 1
        V[2 * p + 2 * blk, base + 7] = 1
        W[4 * blk + 0, base + 7] = 1
        
        # C[0, 2*blk+1] += A[0,2]*B[2, 2*blk+1]
        U[2, base + 8] = 1
        V[2 * p + 2 * blk + 1, base + 8] = 1
        W[4 * blk + 2, base + 8] = 1
        
        # C[1, 2*blk] += A[1,2]*B[2, 2*blk]
        U[5, base + 9] = 1
        V[2 * p + 2 * blk, base + 9] = 1
        W[4 * blk + 1, base + 9] = 1
        
        # C[1, 2*blk+1] += A[1,2]*B[2, 2*blk+1]
        U[5, base + 10] = 1
        V[2 * p + 2 * blk + 1, base + 10] = 1
        W[4 * blk + 3, base + 10] = 1
    
    # Verify the decomposition is correct
    U_arr = U
    V_arr = V
    W_arr = W
    rank = total_rank
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(actual - expected) > 1e-9:
                    # Fallback to schoolbook
                    sb_rank = n * m * p
                    U_sb = np.zeros((n * m, sb_rank))
                    V_sb = np.zeros((m * p, sb_rank))
                    W_sb = np.zeros((p * n, sb_rank))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii * m + jj, col] = 1
                                V_sb[jj * p + kk, col] = 1
                                W_sb[kk * n + ii, col] = 1
                                col += 1
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
