"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map rank index r to (i, j, k)
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # r = i * m * p + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                # U index: i*m + j
                u_idx = i * m + j
                U[u_idx][r] = 1
                
                # V index: j*p + k
                v_idx = j * p + k
                V[v_idx][r] = 1
                
                # W index: k*n + i (trace convention W[k*n+i, r])
                w_idx = k * n + i
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
