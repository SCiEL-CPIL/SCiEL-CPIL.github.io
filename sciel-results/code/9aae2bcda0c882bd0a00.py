"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    # Split p into p1=3 and p2=4
    p1 = 3
    p2 = p - p1
    
    # We will construct a decomposition of rank r1 + r2
    # Using schoolbook for both parts to ensure correctness and simplicity.
    # r1 = n * m * p1 = 3 * 3 * 3 = 9
    # r2 = n * m * p2 = 3 * 3 * 4 = 12
    # Total rank = 21
    
    r1 = n * m * p1
    r2 = n * m * p2
    total_rank = r1 + r2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: A (n x m) * B1 (m x p1) -> C1 (n x p1)
    # B1 corresponds to columns 0 to p1-1 of B
    # C1 corresponds to columns 0 to p1-1 of C
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                # U index: i*m + j
                # V index: j*p + k  (k is 0..2)
                # W index: k*n + i
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    # Part 2: A (n x m) * B2 (m x p2) -> C2 (n x p2)
    # B2 corresponds to columns p1 to p-1 of B
    # C2 corresponds to columns p1 to p-1 of C
    for i in range(n):
        for j in range(m):
            for k_rel in range(p2):
                k_abs = p1 + k_rel
                # U index: i*m + j
                # V index: j*p + k_abs
                # W index: k_abs*n + i
                U[i * m + j][col_idx] = 1
                V[j * p + k_abs][col_idx] = 1
                W[k_abs * n + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
