"""Matrix multiplication 2x4x8 - rank-minimal exact tensor decomposition."""
import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U has shape (4, 7) indexed by [i*2 + j, r]
    V has shape (4, 7) indexed by [j*2 + k, r]
    W has shape (4, 7) indexed by [k*2 + i, r]
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas for C = A*B where A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = M5 - M4 + M1 - M2
    # C[0,1] = M3 + M5
    # C[1,0] = M4 + M6
    # C[1,1] = M5 - M1 + M3 - M7
    # where:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # U: A part, indexed [i*m + j] = [i*2 + j]
    # V: B part, indexed [j*p + k] = [j*2 + k]  
    # W: C part, indexed [k*n + i] = [k*2 + i]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = -1; W[2][1] = -1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = -1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = -1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    deadline = start_time + 120
    
    # Try schoolbook first as fallback
    rank_schoolbook = n * m * p  # 64
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try Strassen-based decomposition
    # We can apply Strassen on 2x2x2 blocks
    # n=2, m=4, p=8
    # Split m into 2 blocks of size 2, p into 4 blocks of size 2
    # Each block is 2x2x2, use Strassen rank 7
    # Total: 2 * 4 = 8 blocks, each rank 7, total rank = 56
    
    if time.time() < deadline - 10:
        U_s, V_s, W_s = _strassen_2x2x2()
        rank_strassen = 8 * 7  # 56
        
        U = [[0] * rank_strassen for _ in range(n * m)]
        V = [[0] * rank_strassen for _ in range(m * p)]
        W = [[0] * rank_strassen for _ in range(p * n)]
        
        for j_block in range(2):  # split m into 2 blocks of 2
            for k_block in range(4):  # split p into 4 blocks of 2
                block_offset = (j_block * 4 + k_block) * 7
                
                # A part: rows i in {0,1}, cols j in {2*j_block, 2*j_block+1}
                # Global A index: i*4 + j_local where j_local = 0 or 1
                # Local A index: i*2 + j_local
                
                # B part: rows j in {2*j_block, 2*j_block+1}, cols k in {2*k_block, 2*k_block+1}
                # Global B index: j_global*8 + k_global
                # Local B index: j_local*2 + k_local
                
                # C part: rows i in {0,1}, cols k in {2*k_block, 2*k_block+1}
                # Global W index: k_global*2 + i
                # Local W index: k_local*2 + i
                
                for r in range(7):
                    for i in range(2):
                        for j_local in range(2):
                            a_idx = i * 2 + j_local
                            if U_s[a_idx][r] != 0:
                                global_a_idx = i * 4 + (2 * j_block + j_local)
                                U[global_a_idx][block_offset + r] += U_s[a_idx][r]
                    
                    for j_local in range(2):
                        for k_local in range(2):
                            b_idx = j_local * 2 + k_local
                            if V_s[b_idx][r] != 0:
                                global_j = 2 * j_block + j_local
                                global_k = 2 * k_block + k_local
                                global_b_idx = global_j * 8 + global_k
                                V[global_b_idx][block_offset + r] += V_s[b_idx][r]
                    
                    for k_local in range(2):
                        for i in range(2):
                            c_idx = k_local * 2 + i
                            if W_s[c_idx][r] != 0:
                                global_k = 2 * k_block + k_local
                                global_w_idx = global_k * 2 + i
                                W[global_w_idx][block_offset + r] += W_s[c_idx][r]
        
        if rank_strassen < best_rank:
            best_rank = rank_strassen
            best_result = {"U": U, "V": V, "W": W}
    
    return best_result

# EVOLVE-BLOCK-END
