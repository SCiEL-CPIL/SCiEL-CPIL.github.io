import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Partition n=3 into n1=2, n2=1
    # Partition m=4 into m1=2, m2=2
    # p=7 remains
    
    n1, n2 = 2, 1
    m1, m2 = 2, 2
    
    # Sub-tensor 1: 2x2x7 (A11, B11)
    # Sub-tensor 2: 2x2x7 (A12, B21)
    # Sub-tensor 3: 1x2x7 (A21, B12)
    # Sub-tensor 4: 1x2x7 (A22, B22)
    
    # We use schoolbook for each sub-tensor to ensure validity.
    # Rank of 2x2x7 schoolbook is 2*2*7 = 28
    # Rank of 1x2x7 schoolbook is 1*2*7 = 14
    # Total Rank = 28 + 28 + 14 + 14 = 84
    
    total_rank = 28 + 28 + 14 + 14
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook decomposition for a block
    def add_block(n_block, m_block, p_block, i_offset, j_offset, k_offset, start_col):
        # i_offset: row offset in A/C
        # j_offset: col offset in A / row offset in B
        # k_offset: col offset in B / col offset in C
        
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    # A index: (i_offset + i, j_offset + j) -> flat (i_offset + i)*m + (j_offset + j)
                    # B index: (j_offset + j, k_offset + k) -> flat (j_offset + j)*p + (k_offset + k)
                    # C index: (k_offset + k, i_offset + i) -> flat (k_offset + k)*n + (i_offset + i)
                    
                    U[(i_offset + i) * m + (j_offset + j)][start_col] = 1
                    V[(j_offset + j) * p + (k_offset + k)][start_col] = 1
                    W[(k_offset + k) * n + (i_offset + i)][start_col] = 1
                    start_col += 1
        return start_col
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:7] -> C[0:7, 0:2]
    col = add_block(n1, m1, p, 0, 0, 0, col)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:7] -> C[0:7, 0:2]
    col = add_block(n1, m2, p, 0, m1, m1, col)
    
    # Block 3: A[2:3, 0:2] * B[0:2, 0:7] -> C[0:7, 2:3]
    col = add_block(n2, m1, p, n1, 0, 0, col)
    
    # Block 4: A[2:3, 2:4] * B[2:4, 0:7] -> C[0:7, 2:3]
    col = add_block(n2, m2, p, n1, m1, m1, col)
    
    return {"U": U, "V": V, "W": W}
