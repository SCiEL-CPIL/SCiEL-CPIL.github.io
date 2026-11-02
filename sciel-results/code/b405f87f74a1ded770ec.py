"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products definition
    # Each product k has:
    # left: list of (block_row, block_col, coeff) for A
    # right: list of (block_row, block_col, coeff) for B
    # out: list of (block_row_c, block_col_a, coeff) for C
    # Note: C block index (c, a) corresponds to output indices k in c, i in a.
    
    products = [
        {
            "left": [(0, 0, 1), (1, 1, 1)],
            "right": [(0, 0, 1), (1, 1, 1)],
            "out": [(0, 0, 1), (1, 1, 1)]
        },
        {
            "left": [(1, 0, 1), (1, 1, 1)],
            "right": [(0, 0, 1)],
            "out": [(1, 0, 1), (1, 1, -1)]
        },
        {
            "left": [(0, 0, 1)],
            "right": [(0, 1, 1), (1, 1, -1)],
            "out": [(0, 1, 1), (1, 1, 1)]
        },
        {
            "left": [(1, 1, 1)],
            "right": [(1, 0, 1), (0, 0, -1)],
            "out": [(0, 0, 1), (1, 0, 1)]
        },
        {
            "left": [(0, 0, 1), (0, 1, 1)],
            "right": [(1, 1, 1)],
            "out": [(0, 0, -1), (0, 1, 1)]
        },
        {
            "left": [(1, 0, 1), (0, 0, -1)],
            "right": [(0, 0, 1), (0, 1, 1)],
            "out": [(1, 1, 1)]
        },
        {
            "left": [(0, 1, 1), (1, 1, -1)],
            "right": [(1, 0, 1), (1, 1, 1)],
            "out": [(0, 0, 1)]
        }
    ]
    
    for k_idx, prod in enumerate(products):
        base_rank = k_idx * 27
        for i_local in range(3):
            for j_local in range(3):
                for k_local in range(3):
                    r = base_rank + (i_local * 9 + j_local * 3 + k_local)
                    
                    # Update U
                    for a, b, coeff in prod["left"]:
                        i_glob = a * 3 + i_local
                        j_glob = b * 3 + j_local
                        U[i_glob * m + j_glob][r] += coeff
                    
                    # Update V
                    for b, c, coeff in prod["right"]:
                        j_glob = b * 3 + j_local
                        k_glob = c * 3 + k_local
                        V[j_glob * p + k_glob][r] += coeff
                    
                    # Update W
                    for c, a, coeff in prod["out"]:
                        k_glob = c * 3 + k_local
                        i_glob = a * 3 + i_local
                        W[k_glob * n + i_glob][r] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
