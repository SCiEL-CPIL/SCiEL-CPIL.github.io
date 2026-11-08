"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    1. Split p=10 into 5 blocks of size 2.
    2. For each 2x3x2 block, use a rank-11 decomposition:
       - Strassen's 7-multiplication algorithm for the 2x2x2 core (using cols 0,1 of A and B).
       - 4 direct multiplications for the remaining 2x1x2 part (using col 2 of A and row 2 of B).
    3. Total rank = 5 * 11 = 55.
    
    This is an improvement over the schoolbook rank of 60.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Pre-define the rank 11 decomposition for M_{2,3,2}
    # Local indices: i in {0,1}, j in {0,1,2}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k_global (k_global = b*2 + k_local)
    # W index: k_global*n + i
    
    # Strassen terms for the 2x2x2 part (j=0,1)
    # Each term is a list of (matrix, index_tuple, value)
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        [
            ('U', (0, 0), 1), ('U', (1, 1), 1),
            ('V', (0, 0), 1), ('V', (1, 1), 1),
            ('W', (0, 0), 1), ('W', (1, 1), 1)
        ],
        # M2: (A10+A11)B00 -> C10, C11
        [
            ('U', (1, 0), 1), ('U', (1, 1), 1),
            ('V', (0, 0), 1),
            ('W', (0, 1), 1), ('W', (1, 1), -1)
        ],
        # M3: A00(B01-B11) -> C01, C11
        [
            ('U', (0, 0), 1),
            ('V', (0, 1), 1), ('V', (1, 1), -1),
            ('W', (1, 0), 1), ('W', (1, 1), 1)
        ],
        # M4: A11(B10-B00) -> C00, C10
        [
            ('U', (1, 1), 1),
            ('V', (1, 0), 1), ('V', (0, 0), -1),
            ('W', (0, 0), 1), ('W', (0, 1), 1)
        ],
        # M5: (A00+A01)B11 -> C00, C01
        [
            ('U', (0, 0), 1), ('U', (0, 1), 1),
            ('V', (1, 1), 1),
            ('W', (0, 0), -1), ('W', (1, 0), 1)
        ],
        # M6: (A10-A00)(B00+B01) -> C11
        [
            ('U', (1, 0), 1), ('U', (0, 0), -1),
            ('V', (0, 0), 1), ('V', (0, 1), 1),
            ('W', (1, 1), 1)
        ],
        # M7: (A01-A11)(B10+B11) -> C00
        [
            ('U', (0, 1), 1), ('U', (1, 1), -1),
            ('V', (1, 0), 1), ('V', (1, 1), 1),
            ('W', (0, 0), 1)
        ]
    ]
    
    # Outer product terms for the 2x1x2 part (j=2)
    outer_terms = [
        # i=0, k=0
        [
            ('U', (0, 2), 1),
            ('V', (2, 0), 1),
            ('W', (0, 0), 1)
        ],
        # i=0, k=1
        [
            ('U', (0, 2), 1),
            ('V', (2, 1), 1),
            ('W', (1, 0), 1)
        ],
        # i=1, k=0
        [
            ('U', (1, 2), 1),
            ('V', (2, 0), 1),
            ('W', (0, 1), 1)
        ],
        # i=1, k=1
        [
            ('U', (1, 2), 1),
            ('V', (2, 1), 1),
            ('W', (1, 1), 1)
        ]
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            # If we run out of time, we return what we have so far,
            # but since the construction is fast, this is unlikely.
            break
        
        for term in all_terms:
            col = col_offset
            for mat_name, idx, val in term:
                if mat_name == 'U':
                    i, j = idx
                    U[i * m + j][col] += val
                elif mat_name == 'V':
                    j, k_local = idx
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                elif mat_name == 'W':
                    k_local, i = idx
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
