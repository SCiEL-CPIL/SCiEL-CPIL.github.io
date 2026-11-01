<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Strassen decomposition for 4x4 matrix multiplication.
    
    Split 4x4 matrices into 2x2 blocks of 2x2 matrices.
    Use Strassen's 7-multiplication scheme at the 4x4 level,
    and schoolbook (4 mults) for each 2x2 sub-multiplication.
    
    Total rank: 7 * 4 = 28
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Recursive Strassen decomposition for 4x4 matrix multiplication.
    
    Split 4x4 matrices into 2x2 blocks of 2x2 matrices.
    Use Strassen's 7-multiplication scheme at the 4x4 level,
    and Strassen's 7-multiplication scheme for each 2x2 sub-multiplication.
    
    Total rank: 7 * 7 = 49
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
>>>>>>> REPLACE

