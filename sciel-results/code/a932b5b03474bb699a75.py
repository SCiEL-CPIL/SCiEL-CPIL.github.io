"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 105
    
    # Start with the trivial decomposition
    best_rank = trivial_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using structured search
    # For matrix multiplication, we can try ranks from some lower bound up to trivial_rank - 1
    
    # Lower bound: at least max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # But realistically, for exact decomposition, we likely need close to n*m*p
    
    # Try a few candidate ranks below 105
    # This is a heuristic search - try to verify if a lower rank is possible
    
    # For now, return the trivial decomposition which is guaranteed correct
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
