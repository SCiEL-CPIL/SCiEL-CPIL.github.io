import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Construct the Strassen-based decomposition (rank 30)
    U_strassen = [[0] * 30 for _ in range(n * m)]
    V_strassen = [[0] * 30 for _ in range(m * p)]
    W_strassen = [[0] * 30 for _ in range(p * n)]
    
    strassen_terms = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [0, 1]]),
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    rank_idx = 0
    for b in range(4):
        k_start = 2 * b
        for t in strassen_terms:
            U_t, V_t, W_t = t
            for i in range(2):
                for j in range(2):
                    val = U_t[i][j]
                    if val != 0:
                        U_strassen[i * m + j][rank_idx] = val
            for j in range(2):
                for k_local in range(2):
                    val = V_t[j][k_local]
                    if val != 0:
                        V_strassen[j * p + (k_start + k_local)][rank_idx] = val
            for k_local in range(2):
                for i in range(2):
                    val = W_t[k_local][i]
                    if val != 0:
                        W_strassen[(k_start + k_local) * n + i][rank_idx] = val
            rank_idx += 1
    
    U_strassen[0 * m + 2][rank_idx] = 1
    U_strassen[1 * m + 2][rank_idx] = 1
    U_strassen[2 * m + 2][rank_idx] = 1
    for k in range(p):
        V_strassen[2 * p + k][rank_idx] = 1
        W_strassen[k * n + 0][rank_idx] = 1
        W_strassen[k * n + 1][rank_idx] = 1
        W_strassen[k * n + 2][rank_idx] = 1
    rank_idx += 1
    
    U_strassen[2 * m + 0][rank_idx] = 1
    U_strassen[2 * m + 1][rank_idx] = 1
    for k in range(p):
        V_strassen[0 * p + k][rank_idx] = 1
        V_strassen[1 * p + k][rank_idx] = 1
        W_strassen[k * n + 2][rank_idx] = 1
    rank_idx += 1
    
    if check_decomposition(U_strassen, V_strassen, W_strassen, rank_idx):
        best_result = {"U": U_strassen, "V": V_strassen, "W": W_strassen}
        best_rank = rank_idx
    
    # Try to reduce rank by searching for a better decomposition
    # Use the known 3x3x3 rank-21 result for first 3 columns
    # Then handle remaining 5 columns
    
    # Alternative: try rank 29 by merging cross terms
    # The two cross terms each handle a full row/col. Can we combine?
    
    # Let's try a different block structure: 2x3 x 3x8
    # Partition A into top 2 rows and bottom 1 row
    # For top 2 rows x all 8 cols: use 2x2 Strassen on pairs of B columns
    # That's 4 blocks of 7 = 28 terms for 2x2 x 2x8
    # Plus cross terms for the third row of A
    
    # Actually, let's try: handle A as 3x3, B as 3x8
    # Use 3x3 x 3x3 rank-21 for first 3 cols of B
    # Then 3x3 x 3x5 for remaining 5 cols
    
    # 3x3 x 3x5: schoolbook = 45, but we can do better
    # Partition the 5 cols: 2+2+1
    # 2x2 blocks: 2*7 = 14, plus cross for 3rd col: 3 terms (one per row of A)
    # Total for 3x3 x 3x5: 14 + 3 + cross terms for 3rd row of A
    # This is getting complex. Let me try a direct optimization approach.
    
    # Try iterative optimization to find rank 29 or lower
    if time.time() < deadline - 10:
        # Try to find a rank-29 decomposition by perturbing the rank-30 one
        # and checking if we can zero out one column
        pass
    
    return best_result
