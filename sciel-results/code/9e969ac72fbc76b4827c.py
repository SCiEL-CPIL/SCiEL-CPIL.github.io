"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin before 120s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 3x4 by 4x7.
    
    Strategy:
    1. The schoolbook decomposition has rank n*m*p = 84.
    2. Strassen-style algorithms typically require square matrices of dimensions that are powers of 2 (or at least even) to exploit recursive block cancellations effectively.
    3. For 3x4 and 4x7, padding to 4x4 and 4x8 (or similar) to use Strassen would result in a rank of at least 49 (for 4x4x4) or higher for larger pads, which is not guaranteed to be less than 84 for the specific non-square case without careful embedding, and often padding overhead dominates.
    4. Specifically, embedding 3x4x7 into a larger square tensor (e.g., 8x8x8) yields rank 343, which is worse.
    5. Therefore, the schoolbook decomposition is the most reliable and likely optimal simple construction for these dimensions.
    
    We return the schoolbook decomposition to ensure validity and exactness.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct U, V, W as lists of lists with integer entries
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
# EVOLVE-BLOCK-END
