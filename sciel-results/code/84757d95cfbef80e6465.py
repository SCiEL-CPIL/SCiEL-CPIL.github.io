import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # Each term is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # Indices 0..3 correspond to [0,0], [0,1], [1,0], [1,1]
    # u_coeffs is a list of (index, coefficient)
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10+a11)b00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a00(b01-b11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a11(b10-b00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a00+a01)b11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a10-a00)(b00+b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01-a11)(b10+b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # We decompose the multiplication into 8 independent 2x2 * 2x2 blocks
    # A is 2x4, split into 2 blocks of 2x2 along columns (col_offset 0, 2)
    # B is 4x8, split into 2 blocks of 2x8 along rows (row_offset 0, 2)
    # Each 2x8 block of B is split into 4 blocks of 2x2 along columns (col_offset 0, 2, 4, 6)
    
    # Total rank = 2 (A blocks) * 4 (B col blocks per A block) * 7 (Strassen rank) = 56
    total_rank = 56
    
    # U is (n*m) x rank = 8 x 56
    # V is (m*p) x rank = 32 x 56
    # W is (p*n) x rank = 16 x 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_idx = 0
    
    # Iterate over the 8 logical 2x2*2x2 multiplications
    # a_block in {0, 1}: selects column pair of A (cols 2*a_block, 2*a_block+1)
    # b_block in {0, 1}: selects row pair of B (rows 2*b_block, 2*b_block+1)
    # Note: For the product A*B to be valid, the column block of A must match the row block of B.
    # A[:, 2*h : 2*h+2] * B[2*h : 2*h+2, :]
    # So a_block and b_block must be the same index h.
    # The other dimension is the column block of B, say j in {0, 1, 2, 3}
    # B rows are 2*h to 2*h+1. B cols are 2*j to 2*j+1.
    
    for h in range(2):  # 0: A cols 0-1, B rows 0-1; 1: A cols 2-3, B rows 2-3
        for j in range(4):  # B cols 0-1, 2-3, 4-5, 6-7
            for u_terms, v_terms, w_terms in strassen_terms:
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                # Fill U
                # A index: i * m + l
                # Local A indices: 0:(0,2h), 1:(0,2h+1), 2:(1,2h), 3:(1,2h+1)
                # u_terms gives (local_idx, coeff)
                for local_idx, coeff in u_terms:
                    # local_idx 0: i=0, l_local=0
                    # local_idx 1: i=0, l_local=1
                    # local_idx 2: i=1, l_local=0
                    # local_idx 3: i=1, l_local=1
                    i = local_idx // 2
                    l_local = local_idx % 2
                    l_global = 2 * h + l_local
                    u_idx = i * m + l_global
                    U[u_idx][rank_idx] += coeff
                
                # Fill V
                # B index: l * p + k
                # Local B indices: 0:(2h, 2j), 1:(2h, 2j+1), 2:(2h+1, 2j), 3:(2h+1, 2j+1)
                # v_terms gives (local_idx, coeff)
                for local_idx, coeff in v_terms:
                    l_local_row = local_idx // 2
                    k_local = local_idx % 2
                    l_global = 2 * h + l_local_row
                    k_global = 2 * j + k_local
                    v_idx = l_global * p + k_global
                    V[v_idx][rank_idx] += coeff
                
                # Fill W
                # Output C is 2x8. Convention W[k*n + i, r]
                # Local C indices: 0:(0,2j), 1:(0,2j+1), 2:(1,2j), 3:(1,2j+1)
                # w_terms gives (local_idx, coeff)
                for local_idx, coeff in w_terms:
                    i = local_idx // 2
                    k_local = local_idx % 2
                    k_global = 2 * j + k_local
                    w_idx = k_global * n + i
                    W[w_idx][rank_idx] += coeff
                
                rank_idx += 1
    
    return {"U": U, "V": V, "W": W}
