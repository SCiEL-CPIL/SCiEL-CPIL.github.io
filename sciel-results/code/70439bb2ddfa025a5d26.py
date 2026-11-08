"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: split p=10 into 5 blocks of size 2.
    # Each block computes a 2x3 @ 3x2 product.
    # Use rank-11 decomposition per block (Strassen 7 + outer 4).
    # Total rank = 55.
    
    num_blocks = 5
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Rank 11 decomposition for M_{2,3,2}
    # A indices: i in {0,1}, j in {0,1,2}
    # B indices: j in {0,1,2}, k in {0,1}
    # C indices: i in {0,1}, k in {0,1}
    # W index convention: k*n + i
    
    # Strassen part (7 terms) for A[:, 0:2] and B[0:2, :]
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)B00 -> C10 +=, C11 -=
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3: A00(B01-B11) -> C01 +=, C11 +=
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        # M4: A11(B10-B00) -> C00 +=, C10 +=
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5: (A00+A01)B11 -> C00 -=, C01 +=
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6: (A10-A00)(B00+B01) -> C11 +=
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11) -> C00 +=
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    # Outer product part (4 terms) for A[:, 2] and B[2, :]
    outer_terms = [
        # i=0, k=0
        {
            'U': [(0,2,1)],
            'V': [(2,0,1)],
            'W': [(0,0,1)]
        },
        # i=0, k=1
        {
            'U': [(0,2,1)],
            'V': [(2,1,1)],
            'W': [(1,0,1)]
        },
        # i=1, k=0
        {
            'U': [(1,2,1)],
            'V': [(2,0,1)],
            'W': [(0,1,1)]
        },
        # i=1, k=1
        {
            'U': [(1,2,1)],
            'V': [(2,1,1)],
            'W': [(1,1,1)]
        }
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col = 0
    for b in range(num_blocks):
        for term in all_terms:
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
