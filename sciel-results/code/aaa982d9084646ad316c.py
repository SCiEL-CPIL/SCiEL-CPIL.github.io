"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy: Use block decomposition splitting p=10 into 5 blocks of size 2.
    Each block is M_{2,3,2} with rank 11 (7 from Strassen for 2x2x2 + 4 for 2x1x2).
    Total rank = 5 * 11 = 55.
    
    The output factor W follows the convention W[k*n+i, r].
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of size 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 55
    
    # Pre-allocate as numpy arrays for speed
    U = np.zeros((n * m, total_rank), dtype=np.int32)
    V = np.zeros((m * p, total_rank), dtype=np.int32)
    W = np.zeros((p * n, total_rank), dtype=np.int32)
    
    # Strassen decomposition for 2x2x2 (7 terms)
    # A: i in {0,1}, j in {0,1}
    # B: j in {0,1}, k in {0,1}
    # C: i in {0,1}, k in {0,1}, W index = k*n + i
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)B00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3: A00(B01-B11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        # M4: A11(B10-B00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5: (A00+A01)B11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6: (A10-A00)(B00+B01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    # Outer product terms for the third column (j=2) of A and third row (j=2) of B
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for term in all_terms:
            col = col_offset
            # Fill U: index i*m + j
            for i, j, val in term['U']:
                U[i * m + j, col] += val
            # Fill V: index j*p + k_global
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global, col] += val
            # Fill W: index k_global*n + i
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i, col] += val
            col_offset += 1
            
    # Convert back to lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
