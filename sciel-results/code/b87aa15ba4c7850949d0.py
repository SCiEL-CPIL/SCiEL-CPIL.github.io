import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, well within 120s/150s
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices as lists of lists for easy manipulation
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    # Define the 4 blocks of 2x2 sub-problems
    # (a_row_start, a_col_start, b_row_start, b_col_start, c_col_start)
    # Note: c_row is always 0:2 for all blocks
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    # Strassen 7 multiplications for 2x2 matrix mult
    # We define them as (u_coeffs, v_coeffs, w_coeffs)
    # Each term is a list of (index, coeff)
    
    # M1 = (u00 + u11) * (v00 + v11)
    # u00, u11 ; v00, v11 -> w00 += 1, w11 += 1
    strassen_terms = [
        # Term 1: (u00 + u11) * (v00 + v11)
        (
            [(0, 1), (3, 1)],  # u00, u11 (local indices 0,1,2,3 -> 00,01,10,11)
            [(0, 1), (3, 1)],  # v00, v11
            [(0, 1), (3, 1)]   # w00, w11
        ),
        # Term 2: (u10 + u11) * v00
        (
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(2, 1), (3, -1)]  # w10 += 1, w11 -= 1
        ),
        # Term 3: u00 * (v01 - v11)
        (
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(1, 1), (3, 1)]   # w01 += 1, w11 += 1
        ),
        # Term 4: u11 * (v10 - v00)
        (
            [(3, 1)],
            [(2, 1), (0, -1)],
            [(0, 1), (2, 1)]   # w00 += 1, w10 += 1
        ),
        # Term 5: (u00 + u01) * v11
        (
            [(0, 1), (1, 1)],
            [(3, 1)],
            [(0, -1), (1, 1)]  # w00 -= 1, w01 += 1
        ),
        # Term 6: (u10 - u00) * (v00 + v01)
        (
            [(2, 1), (0, -1)],
            [(0, 1), (1, 1)],
            [(3, 1)]           # w11 += 1
        ),
        # Term 7: (u01 - u11) * (v10 + v11)
        (
            [(1, 1), (3, -1)],
            [(2, 1), (3, 1)],
            [(0, 1)]           # w00 += 1
        )
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        if time.time() > deadline:
            # In case we hit deadline (shouldn't happen for static code, but good practice)
            break
            
        # Local to Global Index Mapping for this block
        # u local 0->(a_r, a_c), 1->(a_r, a_c+1), 2->(a_r+1, a_c), 3->(a_r+1, a_c+1)
        u_map = [
            u_idx(a_r, a_c),
            u_idx(a_r, a_c + 1),
            u_idx(a_r + 1, a_c),
            u_idx(a_r + 1, a_c + 1)
        ]
        
        # v local 0->(b_r, b_c), 1->(b_r, b_c+1), 2->(b_r+1, b_c), 3->(b_r+1, b_c+1)
        v_map = [
            v_idx(b_r, b_c),
            v_idx(b_r, b_c + 1),
            v_idx(b_r + 1, b_c),
            v_idx(b_r + 1, b_c + 1)
        ]
        
        # w local 0->(0, c_c), 1->(0, c_c+1), 2->(1, c_c), 3->(1, c_c+1)
        w_map = [
            w_idx(0, c_c),
            w_idx(0, c_c + 1),
            w_idx(1, c_c),
            w_idx(1, c_c + 1)
        ]
        
        for (u_terms, v_terms, w_terms) in strassen_terms:
            # Update U
            for local_idx, coeff in u_terms:
                U[u_map[local_idx]][current_rank] += coeff
                
            # Update V
            for local_idx, coeff in v_terms:
                V[v_map[local_idx]][current_rank] += coeff
                
            # Update W
            for local_idx, coeff in w_terms:
                W[w_map[local_idx]][current_rank] += coeff
                
            current_rank += 1
            
    if current_rank != rank:
        # This should not happen
        pass
        
    return {"U": U, "V": V, "W": W}
