import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (32, 28)
    W = [[0] * rank for _ in range(p * n)]  # (16, 28)
    
    # Strassen 2x2x2 patterns
    # U corresponds to A (2x2)
    # V corresponds to B (2x2)
    # W corresponds to C (2x2)
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_strassen_term(base_rank, i_off, j_off, k_off_1, k_off_2):
        """
        Adds a Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * (B[j_off:j_off+2, k_off_1:k_off_1+2] + B[j_off:j_off+2, k_off_2:k_off_2+2])
        """
        for r in range(7):
            current_rank = base_rank + r
            
            # U part (A)
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_off
                global_j = local_j + j_off
                U[global_i * 4 + global_j][current_rank] += val
            
            # V part (B) - Sum of two blocks
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_off
                
                # Add to first block
                global_k_1 = local_k + k_off_1
                V[global_j * 8 + global_k_1][current_rank] += val
                
                # Add to second block
                global_k_2 = local_k + k_off_2
                V[global_j * 8 + global_k_2][current_rank] += val
            
            # W part (C)
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                # The output C is 2x2 for this block, but we need to map it to the global C.
                # The global C column offset should correspond to the "effective" block?
                # Wait. The output C is 2x8.
                # The term computes A * (B_0 + B_1).
                # The result is a 2x2 matrix.
                # But B_0 and B_1 are at different column positions.
                # A * B_0 contributes to C[:, 0:2].
                # A * B_1 contributes to C[:, 2:4].
                # So the W part must be split as well!
                # Strassen computes C = A B.
                # If B = B_0 + B_1, then C = A B_0 + A B_1.
                # The Strassen W vectors give the coefficients for C.
                # We need to add these coefficients to the correct columns of C.
                
                # The local C is 2x2.
                # local_k is the column in the local 2x2 C.
                # If we are processing the sum B_0 + B_1, the result is NOT a single 2x2 block in C.
                # It is two 2x2 blocks in C.
                # So we must add the W contributions to BOTH column offsets.
                
                global_k_1 = local_k + k_off_1
                global_k_2 = local_k + k_off_2
                global_i = local_i + i_off
                
                W[global_k_1 * 2 + global_i][current_rank] += val
                W[global_k_2 * 2 + global_i][current_rank] += val

    # Part 1: A1 (cols 0-1) * B_top (rows 0-1)
    # B_top is split into B_0 (cols 0-1) and B_1 (cols 2-3) for the first Strassen pair
    # and B_2 (cols 4-5) and B_3 (cols 6-7) for the second Strassen pair
    # A1 is at i_off=0, j_off=0
    
    # Term 1: A1 * (B_0 + B_1)
    add_strassen_term(0, 0, 0, 0, 2)
    
    # Term 2: A1 * (B_2 + B_3)
    add_strassen_term(7, 0, 0, 4, 6)
    
    # Part 2: A2 (cols 2-3) * B_bot (rows 2-3)
    # A2 is at i_off=0, j_off=2
    # B_bot is at rows 2-3.
    # Split B_bot into B_4 (cols 0-1), B_5 (cols 2-3), B_6 (cols 4-5), B_7 (cols 6-7)
    
    # Term 3: A2 * (B_4 + B_5)
    add_strassen_term(14, 0, 2, 0, 2)
    
    # Term 4: A2 * (B_6 + B_7)
    add_strassen_term(21, 0, 2, 4, 6)
    
    return {"U": U, "V": V, "W": W}
