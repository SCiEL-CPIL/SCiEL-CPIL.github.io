import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_rank88():
    """
    Construct the rank-88 decomposition for 3x4x8 matrix multiplication.
    Strategy:
    1. Split A (3x4) into A1 (1x4) and A2 (2x4).
    2. A1 * B (1x4 * 4x8) -> Rank 32 (Schoolbook).
    3. A2 * B (2x4 * 4x8) -> Split B into 4 blocks of 4x2? 
       Actually, split j (dim 4) into 2 blocks of 2, and k (dim 8) into 4 blocks of 2.
       This creates 8 sub-problems of size 2x2x2.
       Each 2x2x2 uses Strassen (Rank 7).
       Total Rank for A2 part: 8 * 7 = 56.
    Total Rank: 32 + 56 = 88.
    """
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0, :] (1x4) * B (4x8)
    # Schoolbook: 4 * 8 = 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: A[1:3, :] (2x4) * B (4x8)
    # Split j into 2 blocks of 2: j in {0,1} and {2,3}
    # Split k into 4 blocks of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For each pair (j_block, k_block), we have a 2x2x2 sub-problem.
    # A_sub is 2x2 (rows 1,2 of A; cols j_start, j_start+1)
    # B_sub is 2x2 (rows j_start, j_start+1 of B; cols k_start, k_start+1)
    # C_sub is 2x2 (rows 1,2 of C; cols k_start, k_start+1)
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Local indices for A_sub:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Global i: 1+local_i, j: j_start+local_j
    
    # Local indices for B_sub:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Global j: j_start+local_j, k: k_start+local_k
    
    # Local indices for C_sub:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Global i: 1+local_i, k: k_start+local_k
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                # Fill U
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i = l // 2
                        local_j = l % 2
                        global_i = 1 + local_i
                        global_j = j_start + local_j
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                        
                # Fill V
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j = l // 2
                        local_k = l % 2
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                        
                # Fill W
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i = l // 2
                        local_k = l % 2
                        global_i = 1 + local_i
                        global_k = k_start + local_k
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check if T is all 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    result = _build_rank88()
    # Optional: Verify if time permits, but it's fast
    if _verify(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
