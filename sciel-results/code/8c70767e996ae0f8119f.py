import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 products for 2x2x2 multiplication
    # Each product is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i, j): val}
    # v_coeffs: dict {(j, k): val}
    # w_coeffs: dict {(i, k): val}
    
    products = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # M6
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (0,1): 1}, {(0,0): -1})
    ]
    
    # Two parts: j_base 0 (rows 0,1 of B) and j_base 2 (rows 2,3 of B)
    for j_base in [0, 2]:
        for l in range(4):
            k_start = 2 * l
            for u_coeffs, v_coeffs, w_coeffs in products:
                # Fill U
                for (i, j_loc), val in u_coeffs.items():
                    j_global = j_base + j_loc
                    u_row = i * m + j_global
                    U[u_row][col] += val
                
                # Fill V
                for (j_loc, k_loc), val in v_coeffs.items():
                    j_global = j_base + j_loc
                    k_global = k_start + k_loc
                    v_row = j_global * p + k_global
                    V[v_row][col] += val
                
                # Fill W
                for (i, k_loc), val in w_coeffs.items():
                    k_global = k_start + k_loc
                    w_row = k_global * n + i
                    W[w_row][col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
