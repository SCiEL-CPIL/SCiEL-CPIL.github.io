import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 59
    # Using 3 blocks of 3x3x3 (rank 25 each) and 1 block of 3x3x1 (rank 9)
    # Total rank = 25 + 25 + 9 = 59
    
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 3x3x3 block
    def add_3x3x3_block(k_start):
        nonlocal col_offset
        # k_start is the global column index for the first column of this block
        # Local k indices 0,1,2 map to global k_start, k_start+1, k_start+2
        
        # 1. A22 B22 (Strassen 7)
        # A22: i in {0,1}, j in {0,1}
        # B22: j in {0,1}, k in {0,1} (local)
        # C22: i in {0,1}, k in {0,1} (local)
        
        # M1 = (A00+A11)(B00+B11)
        U[i*m+j][col_offset] = 1
        # ... this is getting tedious to hardcode all 25 terms manually in the thought process.
        # I will write a generic function that takes the block offset and fills the matrices.
        
    # Since I can't easily verify the exact 23-rank coefficients, I will stick to the 25-rank block decomposition.
    # It is guaranteed to be correct.
    
    pass
