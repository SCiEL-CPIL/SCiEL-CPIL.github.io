import numpy as np
import time

def construct():
    """
    Rank-28 Strassen-based decomposition for [2,4,4] matrix multiplication.
    
    The approach uses 4 independent 2x2x2 Strassen blocks (rank 7 each),
    giving total rank 28. This is the known optimal rank for this tensor.
    
    Returns dict with U (8x28), V (16x28), W (8x28) matrices.
    """
    start_time = time.time()
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.int64)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=np.int64)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=np.int64)  # (8, 28)
    
    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    # 4 blocks of 2x2 sub-problems
    # (a_row_start, a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    # Strassen 7 multiplications for 2x2 matrix mult
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # Local indices: 0=00, 1=01, 2=10, 3=11
    
    strassen_terms = [
        # Term 1: (u00 + u11) * (v00 + v11) -> w00 += 1, w11 += 1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # Term 2: (u10 + u11) * v00 -> w10 += 1, w11 -= 1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # Term 3: u00 * (v01 - v11) -> w01 += 1, w11 += 1
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # Term 4: u11 * (v10 - v00) -> w00 += 1, w10 += 1
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # Term 5: (u00 + u01) * v11 -> w00 -= 1, w01 += 1
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # Term 6: (u10 - u00) * (v00 + v01) -> w11 += 1
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # Term 7: (u01 - u11) * (v10 + v11) -> w00 += 1
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        if time.time() > start_time + 100:
            break
            
        # Local to Global Index Mapping
        u_map = [
            u_idx(a_r, a_c),
            u_idx(a_r, a_c + 1),
            u_idx(a_r + 1, a_c),
            u_idx(a_r + 1, a_c + 1)
        ]
        
        v_map = [
            v_idx(b_r, b_c),
            v_idx(b_r, b_c + 1),
            v_idx(b_r + 1, b_c),
            v_idx(b_r + 1, b_c + 1)
        ]
        
        w_map = [
            w_idx(0, c_c),
            w_idx(0, c_c + 1),
            w_idx(1, c_c),
            w_idx(1, c_c + 1)
        ]
        
        for (u_terms, v_terms, w_terms) in strassen_terms:
            for local_idx, coeff in u_terms:
                U[u_map[local_idx], current_rank] += coeff
            for local_idx, coeff in v_terms:
                V[v_map[local_idx], current_rank] += coeff
            for local_idx, coeff in w_terms:
                W[w_map[local_idx], current_rank] += coeff
            current_rank += 1
    
    # Convert to lists for the expected format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
