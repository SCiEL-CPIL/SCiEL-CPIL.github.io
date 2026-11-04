"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x6 matrix.
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, provide the schoolbook decomposition as a fallback
    rank_schoolbook = n * m * p  # 3 * 5 * 6 = 90
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using a structured approach
    # The matrix multiplication tensor T_{i,j,k} = delta_{i',i} delta_{k',k} for (i,j) x (j',k) -> (i',k')
    # We need to find rank-r decomposition where r < n*m*p
    
    # Known lower bounds and known results:
    # For 3x5x6, the Strassen-like or other fast matrix multiplication might help
    # But finding an exact decomposition is hard. Let's try a systematic approach.
    
    # For now, return the schoolbook decomposition to ensure validity
    # A more sophisticated search would be needed to find lower rank, but 
    # the schoolbook rank 90 is guaranteed correct.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
