import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Uses block partition 2+2+1.
    Uses Strassen for 2x2x2 blocks (rank 7 instead of 8).
    Uses schoolbook for other blocks.
    Estimated rank: 117.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Block sizes
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    terms = []
    
    def add_term(u_indices, v_indices, w_indices, coeff=1.0):
        """
        u_indices: list of (global_row, global_col) for A
        v_indices: list of (global_row, global_col) for B
        w_indices: list of (global_k, global_i) for C (k is col of B, i is row of A)
        """
        if coeff == 0:
            return
        # We need to map these to single column entries in U, V, W
        # Since we are building the list of terms first, we store the sparse representation
        terms.append((u_indices, v_indices, w_indices, coeff))
        
    # Helper to get global indices for a block
    def get_block_indices(block_i, block_j, local_r, local_c):
        r = offsets[block_i] + local_r
        c = offsets[block_j] + local_c
        return r, c
        
    # Iterate over all block triples (i, k, j)
    # i: row block of A
    # k: col block of A / row block of B
    # j: col block of B
    for i in range(3):
        for k in range(3):
            for j in range(3):
                s_i = sizes[i]
                s_k = sizes[k]
                s_j = sizes[j]
                
                # Check if this is a 2x2x2 block
                if s_i == 2 and s_k == 2 and s_j == 2:
                    # Use Strassen
                    # Local indices 0,1 for rows/cols
                    # A block is s_i x s_k (2x2)
                    # B block is s_k x s_j (2x2)
                    # C block is s_i x s_j (2x2)
                    
                    # Strassen terms
                    # M1 = (A00+A11)(B00+B11)
                    u_m1 = [get_block_indices(i, k, 0, 0), get_block_indices(i, k, 1, 1)]
                    v_m1 = [get_block_indices(k, j, 0, 0), get_block_indices(k, j, 1, 1)]
                    w_m1 = [ (offsets[j] + 0, offsets[i] + 0), (offsets[j] + 1, offsets[i] + 1) ] # C00, C11
                    # Wait, W index is k_global * n + i_global.
                    # C is i x j.
                    # C_local(0,0) -> i_global = offsets[i]+0, j_global (col of C) = offsets[j]+0.
                    # But W index uses k (col of B) and i (row of A).
                    # k_global = offsets[k] + 0? No.
                    # In the triple (i,k,j), k is the intermediate index.
                    # The output C is indexed by i (row) and j (col).
                    # So C_local(r, c) corresponds to i_global = offsets[i]+r, j_global = offsets[j]+c.
                    # W index: k_global * n + i_global?
                    # The prompt says W[k*n+i, r].
                    # Here k is the column of B. In the triple (i,k,j), k is the column of A and row of B.
                    # So k_global = offsets[k] + k_loc.
                    # i_global = offsets[i] + i_loc.
                    # So w_idx = k_global * 5 + i_global.
                    
                    # Let's re-verify W index mapping.
                    # T[i,j,k] = 1.
                    # U[i,j], V[j,k], W[k,i].
                    # In our loop, i is row of A, k is col of A, j is col of B.
                    # So U index uses i, k.
                    # V index uses k, j.
                    # W index uses j, i. (k in prompt is j in our loop? No.)
                    # Prompt: W[k*n+i].
                    # Standard: C[i,j] = sum_k A[i,k] B[k,j].
                    # So W index corresponds to (j, i).
                    # j is col of B. i is row of A.
                    # In our loop:
                    # i: row of A.
                    # k: col of A.
                    # j: col of B.
                    # So W index should use j and i.
                    # w_idx = j_global * n + i_global.
                    
                    # Correct W mapping:
                    # w_idx = (offsets[j] + c_local) * 5 + (offsets[i] + r_local)
                    
                    w_m1 = [ (offsets[j] + 0) * 5 + (offsets[i] + 0), (offsets[j] + 1) * 5 + (offsets[i] + 1) ]
                    add_term(u_m1, v_m1, w_m1, 1.0)
                    
                    # M2 = (A10+A11)B00
                    u_m2 = [get_block_indices(i, k, 1, 0), get_block_indices(i, k, 1, 1)]
                    v_m2 = [get_block_indices(k, j, 0, 0)]
                    w_m2 = [ (offsets[j] + 0) * 5 + (offsets[i] + 1), (offsets[j] + 1) * 5 + (offsets[i] + 1) ] # C10, C11
                    # C10: i=1, j=0. C11: i=1, j=1.
                    add_term(u_m2, v_m2, w_m2, 1.0)
                    
                    # M3 = A00(B01-B11)
                    u_m3 = [get_block_indices(i, k, 0, 0)]
                    v_m3 = [get_block_indices(k, j, 0, 1), get_block_indices(k, j, 1, 1)]
                    w_m3 = [ (offsets[j] + 0) * 5 + (offsets[i] + 0), (offsets[j] + 1) * 5 + (offsets[i] + 0) ] # C00, C01
                    # C00: i=0, j=0. C01: i=0, j=1.
                    add_term(u_m3, v_m3, w_m3, 1.0)
                    # Note: M3 has -B11. So coeff for B11 part is -1.
                    # We can split this into two terms or handle signed coefficients.
                    # The interface allows integer/half-integer entries.
                    # So we can have coeff -1 in V.
                    # But add_term takes a single coeff for the whole term.
                    # We need to support vectors with different coefficients.
                    
                    # Let's refactor add_term to accept lists of (index, coeff)
                    pass

# EVOLVE-BLOCK-END
