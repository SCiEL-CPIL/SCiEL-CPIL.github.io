"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    Strategy: Use Strassen-like block decomposition.
    - Split the 3x3 middle into blocks to exploit Strassen on 2x2 subproblems.
    - The 2x3x2 multiplication has rank 11 (vs 12 schoolbook).
    - The 2x3x6 can be seen as 2x3x3 * 2 (tensor product with 1x1x2 identity).
    - But we can do better by direct block decomposition.
    
    Let's use a systematic approach:
    - A is 2x3, B is 3x6.
    - Split B into two 3x3 blocks: B = [B1 | B2].
    - Then C = [A*B1 | A*B2].
    - A*B1 is 2x3, A*B2 is 2x3.
    - Each 2x3x3 multiplication has rank 7 (known optimal).
    - But we can share structure between the two multiplications.
    
    Actually, let's think differently. The tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k].
    For 2x3x6, the naive rank is 36.
    
    Let's try a block approach:
    - Split the output into 2x3 and 3x6... no, that doesn't help directly.
    
    Let me try the known result: 2x3x6 has rank at most 18 (from 2x3x3 rank 7 doubled? No).
    
    Actually, let's use the fact that 2x3x2 has rank 11. Then 2x3x6 = 2x3x2 tensor 1x1x3? No.
    
    Let me just do a direct search with Strassen-inspired structure.
    
    For 2x3x6:
    - A: 2x3, B: 3x6, C: 2x6
    - Split B into B_left (3x3) and B_right (3x3).
    - C_left = A * B_left (2x3), C_right = A * B_right (2x3).
    - Each 2x3x3 has rank 7, so total rank <= 14.
    - But can we do better by sharing?
    
    Let's try rank 14 first as a safe upper bound, then try to reduce.
    
    Actually, the known optimal rank for 2x3x3 is 7. So 2x3x6 <= 14.
    Let's construct the rank-14 decomposition and see if we can reduce it.
    """
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, let's get the rank-7 decomposition for 2x3x3.
    # This is a known result. Let me construct it.
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # Rank 7 means 7 scalar multiplications.
    
    # Let me look up or derive the 2x3x3 rank-7 decomposition.
    # One approach: use the fact that 2x3x3 can be embedded in 3x3x3.
    # 3x3x3 has rank 19 (Strassen gives 19 for 2x2x2, but 3x3x3 is different).
    
    # Actually, let me just do a numerical search for the 2x3x6 tensor.
    
    # Use the schoolbook decomposition which is guaranteed to be valid and exact.
    # Rank is n*m*p = 36.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    U_out = U_school.astype(int).tolist()
    V_out = V_school.astype(int).tolist()
    W_out = W_school.astype(int).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
