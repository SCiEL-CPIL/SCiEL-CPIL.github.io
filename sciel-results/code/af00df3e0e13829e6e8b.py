import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    # We will build the decomposition with rank 108.
    # 1. 4x4x4 part (k=0..3): 8 Strassen 2x2x2 blocks. Rank 56.
    # 2. 4x1x4 part (k=4, j=4? No, j is 0..4. Wait.)
    # Let's re-verify the splitting.
    # A is 4x5. B is 5x6.
    # C is 4x6.
    
    # Split B's columns (k) into 0..3 and 4..5.
    # Part 1: k in 0..3. This is C[:, 0:4] = A @ B[:, 0:4].
    # This is a 4x5 @ 5x4 multiplication.
    # Split A's rows (i) and B's rows (j)?
    # A is 4x5. B_part is 5x4.
    # We can split this 4x5x4 into:
    # - 4x4x4 part (using j=0..3 of B_part? No, j is the inner dimension).
    #   A[:, 0:4] @ B_part[0:4, :] is 4x4 @ 4x4.
    #   A[:, 4:5] @ B_part[4:5, :] is 4x1 @ 1x4.
    #   
    #   So Part 1 (4x5x4) splits into:
    #   1a. 4x4x4 (A cols 0-3, B rows 0-3).
    #   1b. 4x1x4 (A col 4, B row 4).
    
    # Part 2: k in 4..5. This is C[:, 4:6] = A @ B[:, 4:6].
    # This is a 4x5 @ 5x2 multiplication.
    # Split A's rows (i) into 0..1 and 2..3.
    # 2a. 2x5x2 (A rows 0-1).
    # 2b. 2x5x2 (A rows 2-3).
    
    # Ranks:
    # 1a. 4x4x4: 8 Strassen 2x2x2 blocks. Rank 56.
    # 1b. 4x1x4: Schoolbook. Rank 16.
    # 2a. 2x5x2: Split j into 0-1, 2, 3-4.
    #       2x2x2 (Strassen, 7) + 2x1x2 (Schoolbook, 4) + 2x2x2 (Strassen, 7) = 18.
    # 2b. 2x5x2: Same, Rank 18.
    
    # Total Rank = 56 + 16 + 18 + 18 = 108.

    RANK = 108
    U = np.zeros((N, RANK))
    V = np.zeros((M, RANK))
    W = np.zeros((P, RANK))
    
    col = 0

    def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    def add_strassen_block(i_off, j_off, k_off):
        # Adds a 2x2x2 Strassen block.
        # Local indices i_l, j_l, k_l in {0, 1}
        # Global indices:
        # i = i_off + i_l
        # j = j_off + j_l
        # k = k_off + k_l
        
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # 7 terms
        # Term 1: (a00+a11)(b00+b11) -> c00, c11
        # u: (0,0), (1,1)
        # v: (0,0), (1,1)
        # w: (0,0), (1,1)
        
        # We need to add 7 rank-1 terms? No, Strassen gives 7 products.
        # Each product corresponds to ONE rank-1 term in the tensor decomposition?
        # Yes, if we define the vectors correctly.
        
        # Let's define the 7 vectors for U, V, W for this block.
        # Then add them.
        
        # Term 1
        u_idx = [i_off * m + (j_off + 0), (i_off + 1) * m + (j_off + 1)]
        v_idx = [(j_off + 0) * p + (k_off + 0), (j_off + 1) * p + (k_off + 1)]
        w_idx = [(k_off + 0) * n + i_off, (k_off + 1) * n + (i_off + 1)]
        
        # Wait, the indices in u_idx, v_idx, w_idx are for the non-zero elements.
        # For Term 1:
        # U is non-zero at (i_off, j_off) and (i_off+1, j_off+1)
        # V is non-zero at (j_off, k_off) and (j_off+1, k_off+1)
        # W is non-zero at (k_off, i_off) and (k_off+1, i_off+1)
        
        # But add_rank1 only takes one index.
        # So we need to call add_rank1 multiple times?
        # No, a rank-1 term in the tensor is a single column in U, V, W.
        # The column can have multiple non-zero entries.
        # So we should modify U, V, W directly for this column.
        
        def set_col(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            nonlocal col
            for idx, val in zip(u_indices, u_vals):
                U[idx, col] = val
            for idx, val in zip(v_indices, v_vals):
                V[idx, col] = val
            for idx, val in zip(w_indices, w_vals):
                W[idx, col] = val
            col += 1

        # Term 1: M1
        set_col(
            [(i_off)*m + (j_off+0), (i_off+1)*m + (j_off+1)], [1, 1],
            [(j_off+0)*p + (k_off+0), (j_off+1)*p + (k_off+1)], [1, 1],
            [(k_off+0)*n + (i_off+0), (k_off+1)*n + (i_off+1)], [1, 1]
        )
        
        # Term 2: M2 = (a10+a11)b00 -> c10
        set_col(
            [(i_off+1)*m + (j_off+0), (i_off+1)*m + (j_off+1)], [1, 1],
            [(j_off+0)*p + (k_off+0)], [1],
            [(k_off+0)*n + (i_off+1)], [1]
        )
        
        # Term 3: M3 = a00(b01-b11) -> c01
        set_col(
            [(i_off+0)*m + (j_off+0)], [1],
            [(j_off+0)*p + (k_off+1), (j_off+1)*p + (k_off+1)], [1, -1],
            [(k_off+1)*n + (i_off+0)], [1]
        )
        
        # Term 4: M4 = a11(b10-b00) -> c00, c10
        set_col(
            [(i_off+1)*m + (j_off+1)], [1],
            [(j_off+1)*p + (k_off+0), (j_off+0)*p + (k_off+0)], [1, -1],
            [(k_off+0)*n + (i_off+0), (k_off+0)*n + (i_off+1)], [1, 1]
        )
        
        # Term 5: M5 = (a00+a01)b11 -> c00, c01
        set_col(
            [(i_off+0)*m + (j_off+0), (i_off+0)*m + (j_off+1)], [1, 1],
            [(j_off+1)*p + (k_off+1)], [1],
            [(k_off+0)*n + (i_off+0), (k_off+1)*n + (i_off+0)], [-1, 1]
        )
        # Note: c00 gets -M5, c01 gets +M5.
        # So W values: -1 for c00, 1 for c01.
        
        # Term 6: M6 = (a10-a00)(b00+b01) -> c11
        set_col(
            [(i_off+1)*m + (j_off+0), (i_off+0)*m + (j_off+0)], [1, -1],
            [(j_off+0)*p + (k_off+0), (j_off+0)*p + (k_off+1)], [1, 1],
            [(k_off+1)*n + (i_off+1)], [1]
        )
        
        # Term 7: M7 = (a01-a11)(b10+b11) -> c00, c11
        set_col(
            [(i_off+0)*m + (j_off+1), (i_off+1)*m + (j_off+1)], [1, -1],
            [(j_off+1)*p + (k_off+0), (j_off+1)*p + (k_off+1)], [1, 1],
            [(k_off+0)*n + (i_off+0), (k_off+1)*n + (i_off+1)], [1, 1]
        )

    def add_schoolbook_block(i_off, j_off, k_off, n_b, m_b, p_b):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    i_g = i_off + i
                    j_g = j_off + j
                    k_g = k_off + k
                    u_idx = i_g * m + j_g
                    v_idx = j_g * p + k_g
                    w_idx = k_g * n + i_g
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

    # 1a. 4x4x4 part (i=0..3, j=0..3, k=0..3)
    # 8 blocks of 2x2x2
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(2):
                add_strassen_block(i_block*2, j_block*2, k_block*2)
    
    # 1b. 4x1x4 part (i=0..3, j=4, k=0..3)
    add_schoolbook_block(0, 4, 0, 4, 1, 4)
    
    # 2a. 2x5x2 part (i=0..1, j=0..4, k=4..5)
    # Split j into 0-1, 2, 3-4
    # j=0-1: 2x2x2 Strassen
    add_strassen_block(0, 0, 4)
    # j=2: 2x1x2 Schoolbook
    add_schoolbook_block(0, 2, 4, 2, 1, 2)
    # j=3-4: 2x2x2 Strassen
    add_strassen_block(0, 3, 4)
    
    # 2b. 2x5x2 part (i=2..3, j=0..4, k=4..5)
    # j=0-1: 2x2x2 Strassen
    add_strassen_block(2, 0, 4)
    # j=2: 2x1x2 Schoolbook
    add_schoolbook_block(2, 2, 4, 2, 1, 2)
    # j=3-4: 2x2x2 Strassen
    add_strassen_block(2, 3, 4)
    
    if col != RANK:
        # If col > RANK, we have an error in rank calculation.
        # If col < RANK, we have unused space.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
