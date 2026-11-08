import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Generate all triples (i, j, k) and assign to unique columns
    # The column index c corresponds to the linear index of (i, j, k)
    # We iterate in a specific order to fill the columns
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row is i*m + j
                # V row is j*p + k
                # W row is k*n + i
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row][col_idx] = 1.0
                V[v_row][col_idx] = 1.0
                W[w_row][col_idx] = 1.0
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
