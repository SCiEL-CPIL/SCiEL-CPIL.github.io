import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # n=4 -> [0,1], [2,3]
    # m=5 -> [0,1], [2,3], [4]
    # p=6 -> [0,1], [2,3], [4,5]
    
    i_blocks = [(0, 2), (2, 4)]
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    U = []
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (i, j)
        # v_indices: list of (j, k)
        # w_indices: list of (k, i) -- Note: W index is k*n+i
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for (i, j) in u_indices:
            col_u[i * m + j] = coeff
        for (j, k) in v_indices:
            col_v[j * p + k] = coeff
        for (k, i) in w_indices:
            col_w[k * n + i] = coeff
            
        U.append(col_u)
        V.append(col_v)
        W.append(col_w)

    # Strassen terms for 2x2x2x2 blocks
    # A is 2x2, B is 2x2
    # Local indices 0,1 for rows/cols of blocks
    def strassen_block(i_start, j_start, k_start):
        # i_start, j_start, k_start are starting indices
        # Block size 2x2x2x2
        # A[i_start:i_start+2, j_start:j_start+2]
        # B[j_start:j_start+2, k_start:k_start+2]
        # C[i_start:i_start+2, k_start:k_start+2]
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            [(i_start, j_start), (i_start+1, j_start+1)],
            [(j_start, k_start), (j_start+1, k_start+1)],
            [(k_start, i_start), (k_start, i_start+1), (k_start+1, i_start), (k_start+1, i_start+1)]
        )
        
        # M2 = (A10 + A11)B00
        add_term(
            [(i_start+1, j_start), (i_start+1, j_start+1)],
            [(j_start, k_start)],
            [(k_start, i_start+1), (k_start+1, i_start+1)]
        )
        
        # M3 = A00(B01 - B11)
        add_term(
            [(i_start, j_start)],
            [(j_start, k_start+1)],
            [(k_start+1, i_start)]
        )
        add_term(
            [(i_start, j_start)],
            [(j_start+1, k_start+1)],
            [(k_start+1, i_start)],
            coeff=-1
        )
        
        # M4 = A11(B10 - B00)
        add_term(
            [(i_start+1, j_start+1)],
            [(j_start+1, k_start)],
            [(k_start, i_start+1)]
        )
        add_term(
            [(i_start+1, j_start+1)],
            [(j_start, k_start)],
            [(k_start, i_start+1)],
            coeff=-1
        )
        
        # M5 = (A00 + A01)B11
        add_term(
            [(i_start, j_start), (i_start, j_start+1)],
            [(j_start+1, k_start+1)],
            [(k_start+1, i_start)]
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            [(i_start+1, j_start)],
            [(j_start, k_start), (j_start, k_start+1)],
            [(k_start, i_start+1), (k_start+1, i_start+1)]
        )
        add_term(
            [(i_start, j_start)],
            [(j_start, k_start), (j_start, k_start+1)],
            [(k_start, i_start+1), (k_start+1, i_start+1)],
            coeff=-1
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            [(i_start, j_start+1)],
            [(j_start+1, k_start), (j_start+1, k_start+1)],
            [(k_start, i_start), (k_start+1, i_start)]
        )
        add_term(
            [(i_start+1, j_start+1)],
            [(j_start+1, k_start), (j_start+1, k_start+1)],
            [(k_start, i_start), (k_start+1, i_start)],
            coeff=-1
        )

    # Schoolbook terms for 2x1x1x2 blocks
    # A is 2x1, B is 1x2
    def schoolbook_block(i_start, j_start, k_start):
        # A[i_start:i_start+2, j_start]
        # B[j_start, k_start:k_start+2]
        # C[i_start:i_start+2, k_start:k_start+2]
        
        add_term(
            [(i_start, j_start)],
            [(j_start, k_start)],
            [(k_start, i_start)]
        )
        add_term(
            [(i_start, j_start)],
            [(j_start, k_start+1)],
            [(k_start+1, i_start)]
        )
        add_term(
            [(i_start+1, j_start)],
            [(j_start, k_start)],
            [(k_start, i_start+1)]
        )
        add_term(
            [(i_start+1, j_start)],
            [(j_start, k_start+1)],
            [(k_start+1, i_start+1)]
        )

    # Iterate over blocks
    for i_idx, (i_start, i_end) in enumerate(i_blocks):
        for j_idx, (j_start, j_end) in enumerate(j_blocks):
            for k_idx, (k_start, k_end) in enumerate(k_blocks):
                # Determine block sizes
                h_i = i_end - i_start
                h_j = j_end - j_start
                h_k = k_end - k_start
                
                if h_i == 2 and h_j == 2 and h_k == 2:
                    strassen_block(i_start, j_start, k_start)
                elif h_i == 2 and h_j == 1 and h_k == 2:
                    schoolbook_block(i_start, j_start, k_start)
                else:
                    # Fallback to schoolbook for any other combination (should not happen)
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                add_term(
                                    [(i, j)],
                                    [(j, k)],
                                    [(k, i)]
                                )

    return {"U": U, "V": V, "W": W}
