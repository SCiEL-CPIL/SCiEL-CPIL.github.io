import time
_start_time = time.time()
_TIMEOUT = 145.0  # safety margin before 150s kill

def construct():
    n, m, p = 2, 4, 8
    
    # Strassen 2x2 local coefficients
    # Format: (u_idx, v_idx, w_idx, coeff)
    # u_idx, v_idx in 0..3 (local 2x2)
    # w_idx in 0..3 (local 2x2)
    # coeff is 1 or -1
    # Standard Strassen:
    # M0 = (a00+a11)(b00+b11) -> c00+1, c11+1
    # M1 = (a10+a11)b00 -> c10+1, c11-1
    # M2 = a00(b01-b11) -> c01+1, c11+1
    # M3 = a11(b10-b00) -> c00+1, c10+1
    # M4 = (a00+a01)b11 -> c00-1, c01+1
    # M5 = (a10-a00)(b00+b01) -> c11+1
    # M6 = (a01-a11)(b10+b11) -> c00+1
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices, coeffs)
        # Each term is a rank-1 tensor contribution
        # We need to map local indices to global indices.
        
        # r=0
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # r=1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # r=2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # r=3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # r=4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # r=5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # r=6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1
            
            # Local to Global Maps
            # A: 2x4. Row i, Col j. Local 2x2.
            # half 0: Cols 0,1. half 1: Cols 2,3.
            a_map = {
                0: 0 * m + (2 * half + 0),
                1: 0 * m + (2 * half + 1),
                2: 1 * m + (2 * half + 0),
                3: 1 * m + (2 * half + 1)
            }
            
            # B: 4x8. Row j, Col k. Local 2x2.
            # half 0: Rows 0,1. half 1: Rows 2,3.
            # jj 0..3: Cols 2*jj, 2*jj+1
            b_map = {
                0: (2 * half + 0) * p + (2 * jj + 0),
                1: (2 * half + 0) * p + (2 * jj + 1),
                2: (2 * half + 1) * p + (2 * jj + 0),
                3: (2 * half + 1) * p + (2 * jj + 1)
            }
            
            # C: 2x8. Row i, Col k. Local 2x2.
            # W index: k * n + i
            c_map = {
                0: (2 * jj + 0) * n + 0,
                1: (2 * jj + 1) * n + 0,
                2: (2 * jj + 0) * n + 1,
                3: (2 * jj + 1) * n + 1
            }

            for r in range(7):
                rr = base_r + r
                u_list, v_list, w_list = strassen_terms[r]
                
                for idx, val in u_list:
                    U[a_map[idx]][rr] += val
                    
                for idx, val in v_list:
                    V[b_map[idx]][rr] += val
                    
                for idx, val in w_list:
                    W[c_map[idx]][rr] += val

    return {"U": U, "V": V, "W": W}
