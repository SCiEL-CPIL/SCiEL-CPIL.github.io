import numpy as np
import time

DIMENSIONS = [2, 4, 7]
START_TIME = time.time()
DEADLINE = 145.0 # 5s margin

def construct():
    # Check deadline
    if time.time() > DEADLINE:
        pass

    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing contributions from blocks.
    # The total rank will be the sum of the ranks of the blocks.
    # We use a list of tuples (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals) for each rank-1 term
    # to accumulate them, then convert to the matrix format.
    
    terms = []
    
    def add_term(u_map, v_map, w_map):
        # u_map, v_map, w_map are dicts: {index: value}
        terms.append((u_map, v_map, w_map))

    # Strassen 2x2x2 block decomposition
    # Inputs: A[2,2], B[2,2] -> C[2,2]
    # Local indices:
    # A: 0,1 (row 0), 2,3 (row 1)
    # B: 0,1 (col 0), 2,3 (col 1)
    # C: 0,1 (row 0), 2,3 (row 1)
    
    def add_strassen_block(a_base, b_base, c_base):
        # a_base, b_base, c_base are the base indices in the global U, V, W matrices
        # A indices: a_base + [0, 1, 2, 3]
        # B indices: b_base + [0, 1, 2, 3]
        # C indices: c_base + [0, 1, 2, 3]
        
        a0, a1, a2, a3 = a_base + [0, 1, 2, 3]
        b0, b1, b2, b3 = b_base + [0, 1, 2, 3]
        c0, c1, c2, c3 = c_base + [0, 1, 2, 3]
        
        # M1 = (a00 + a11) * (b00 + b11)
        add_term({a0: 1, a3: 1}, {b0: 1, b3: 1}, {c0: 1, c3: 1})
        
        # M2 = (a10 + a11) * b00
        add_term({a2: 1, a3: 1}, {b0: 1}, {c2: 1, c3: -1})
        
        # M3 = a00 * (b10 - b11)
        add_term({a0: 1}, {b2: 1, b3: -1}, {c1: 1, c3: 1})
        
        # M4 = a11 * (b01 - b00)
        add_term({a3: 1}, {b1: 1, b0: -1}, {c0: 1, c2: 1})
        
        # M5 = (a00 + a01) * b11
        add_term({a0: 1, a1: 1}, {b3: 1}, {c0: -1, c1: 1})
        
        # M6 = (a10 - a00) * (b00 + b01)
        add_term({a2: 1, a0: -1}, {b0: 1, b1: 1}, {c3: 1})
        
        # M7 = (a01 - a11) * (b01 + b11)
        add_term({a1: 1, a3: -1}, {b1: 1, b3: 1}, {c0: 1})

    # Schoolbook 2x2x1 block decomposition
    # Inputs: A[2,2], B[2,1] -> C[2,1]
    # Local indices:
    # A: 0,1 (row 0), 2,3 (row 1)
    # B: 0,1 (col 0)
    # C: 0,1 (row 0, row 1)
    
    def add_schoolbook_221_block(a_base, b_base, c_base):
        a0, a1, a2, a3 = a_base + [0, 1, 2, 3]
        b0, b1 = b_base + [0, 1]
        c0, c1 = c_base + [0, 1]
        
        # P1 = a00 * b00
        add_term({a0: 1}, {b0: 1}, {c0: 1})
        
        # P2 = a01 * b01
        add_term({a1: 1}, {b1: 1}, {c0: 1})
        
        # P3 = a10 * b00
        add_term({a2: 1}, {b0: 1}, {c1: 1})
        
        # P4 = a11 * b01
        add_term({a3: 1}, {b1: 1}, {c1: 1})

    # Global Indices:
    # U (A): size n*m = 2*4 = 8. Index = i*m + j. i in [0,1], j in [0,3].
    # V (B): size m*p = 4*7 = 28. Index = j*p + k. j in [0,3], k in [0,6].
    # W (C): size p*n = 7*2 = 14. Index = k*n + i. k in [0,6], i in [0,1].
    
    # Decomposition Strategy:
    # Split the inner dimension m=4 into two blocks of size 2: j in [0,1] and j in [2,3].
    # For each block, we have a 2x2 (A_block) * 2x7 (B_block) -> 2x7 (C_block) multiplication.
    # We can further split the output dimension p=7 into blocks of size 2,2,2,1.
    # For each (A_block, B_block_pair) we use Strassen (rank 7) for 2x2x2, and Schoolbook (rank 4) for 2x2x1.
    
    # Block 1: j in [0,1]
    # A indices: 0,1 (i=0), 2,3 (i=1) -> base 0
    # B indices:
    #   k=0,1: j=0 -> 0*7+0=0, j=1 -> 1*7+0=7. So B indices for k=0 are 0,7. For k=1 are 1,8.
    #   Wait, V index is j*p + k.
    #   For a 2x2 block of B covering rows j=0,1 and cols k=0,1:
    #   The elements are V[0,0], V[0,1], V[1,0], V[1,1].
    #   Global indices:
    #   V[0,0] -> 0*7 + 0 = 0
    #   V[0,1] -> 0*7 + 1 = 1
    #   V[1,0] -> 1*7 + 0 = 7
    #   V[1,1] -> 1*7 + 1 = 8
    #   These are NOT contiguous in the V matrix layout (j*p+k).
    #   However, our `add_term` takes explicit indices. We just need to map the local Strassen indices to global V indices.
    
    # Let's define a helper to get global V index for a block
    def get_v_indices(j_start, k_start):
        # Returns [v00, v01, v10, v11] for B block starting at row j_start, col k_start
        # v00 corresponds to B[j_start, k_start]
        # v01 corresponds to B[j_start, k_start+1]
        # v10 corresponds to B[j_start+1, k_start]
        # v11 corresponds to B[j_start+1, k_start+1]
        return [
            j_start * p + k_start,
            j_start * p + (k_start + 1),
            (j_start + 1) * p + k_start,
            (j_start + 1) * p + (k_start + 1)
        ]
    
    def get_w_indices(k_start, i_start=0):
        # Returns [w00, w01, w10, w11] for C block starting at col k_start, row i_start
        # w00 corresponds to C[k_start, i_start]
        # w01 corresponds to C[k_start+1, i_start]
        # w10 corresponds to C[k_start, i_start+1]
        # w11 corresponds to C[k_start+1, i_start+1]
        return [
            k_start * n + i_start,
            (k_start + 1) * n + i_start,
            k_start * n + (i_start + 1),
            (k_start + 1) * n + (i_start + 1)
        ]

    # We need to adapt add_strassen_block and add_schoolbook_221_block to take explicit index lists
    # instead of base indices, because the indices are not contiguous.
    
    def add_strassen_block_explicit(a_idx, b_idx, c_idx):
        # a_idx, b_idx, c_idx are lists of 4 indices
        # Local 0,1,2,3 map to a_idx[0..3] etc.
        
        # M1 = (a0 + a3) * (b0 + b3)
        add_term({a_idx[0]: 1, a_idx[3]: 1}, {b_idx[0]: 1, b_idx[3]: 1}, {c_idx[0]: 1, c_idx[3]: 1})
        
        # M2 = (a2 + a3) * b0
        add_term({a_idx[2]: 1, a_idx[3]: 1}, {b_idx[0]: 1}, {c_idx[2]: 1, c_idx[3]: -1})
        
        # M3 = a0 * (b2 - b3)
        add_term({a_idx[0]: 1}, {b_idx[2]: 1, b_idx[3]: -1}, {c_idx[1]: 1, c_idx[3]: 1})
        
        # M4 = a3 * (b1 - b0)
        add_term({a_idx[3]: 1}, {b_idx[1]: 1, b_idx[0]: -1}, {c_idx[0]: 1, c_idx[2]: 1})
        
        # M5 = (a0 + a1) * b3
        add_term({a_idx[0]: 1, a_idx[1]: 1}, {b_idx[3]: 1}, {c_idx[0]: -1, c_idx[1]: 1})
        
        # M6 = (a2 - a0) * (b0 + b1)
        add_term({a_idx[2]: 1, a_idx[0]: -1}, {b_idx[0]: 1, b_idx[1]: 1}, {c_idx[3]: 1})
        
        # M7 = (a1 - a3) * (b1 + b3)
        add_term({a_idx[1]: 1, a_idx[3]: -1}, {b_idx[1]: 1, b_idx[3]: 1}, {c_idx[0]: 1})

    def add_schoolbook_221_block_explicit(a_idx, b_idx, c_idx):
        # a_idx: 4 indices
        # b_idx: 2 indices
        # c_idx: 2 indices
        
        # P1 = a0 * b0
        add_term({a_idx[0]: 1}, {b_idx[0]: 1}, {c_idx[0]: 1})
        
        # P2 = a1 * b1
        add_term({a_idx[1]: 1}, {b_idx[1]: 1}, {c_idx[0]: 1})
        
        # P3 = a2 * b0
        add_term({a_idx[2]: 1}, {b_idx[0]: 1}, {c_idx[1]: 1})
        
        # P4 = a3 * b1
        add_term({a_idx[3]: 1}, {b_idx[1]: 1}, {c_idx[1]: 1})

    # Main Loop
    # j_start in [0, 2]
    for j_start in [0, 2]:
        # A indices for this block (rows i=0,1, cols j=j_start, j_start+1)
        a_idx = [
            0 * m + j_start,      # i=0, j=j_start
            0 * m + (j_start+1),  # i=0, j=j_start+1
            1 * m + j_start,      # i=1, j=j_start
            1 * m + (j_start+1)   # i=1, j=j_start+1
        ]
        
        # k_start in [0, 2, 4] for Strassen (2x2x2)
        for k_start in [0, 2, 4]:
            b_idx = get_v_indices(j_start, k_start)
            c_idx = get_w_indices(k_start, 0)
            add_strassen_block_explicit(a_idx, b_idx, c_idx)
            
        # k_start = 6 for Schoolbook (2x2x1)
        k_start = 6
        b_idx = [
            j_start * p + k_start,
            (j_start + 1) * p + k_start
        ]
        c_idx = [
            k_start * n + 0,
            k_start * n + 1
        ]
        add_schoolbook_221_block_explicit(a_idx, b_idx, c_idx)

    rank = len(terms)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_map, v_map, w_map) in enumerate(terms):
        for idx, val in u_map.items():
            U[idx][col] = val
        for idx, val in v_map.items():
            V[idx][col] = val
        for idx, val in w_map.items():
            W[idx][col] = val

    return {"U": U, "V": V, "W": W}
