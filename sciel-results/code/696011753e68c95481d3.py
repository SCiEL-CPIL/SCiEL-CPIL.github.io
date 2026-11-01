"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 construction
    rank = 2 * p  # 2 * 14 = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Rank index: s*p + k' where s in {0, 1}, k' in {0, ..., p-1}
    
    for i in range(n):
        for j in range(m):
            for s in range(2):
                for kp in range(p):
                    r = s * p + kp
                    # U[i*m+j, r] is 1 if s == 0
                    if s == 0:
                        U[i * m + j][r] = 1
                    
                    # V[j*p+k, r] is 1 if k == kp
                    # We iterate over k in the main loop? 
                    # It's easier to fill V and W by iterating over their row indices.
                    
    # Fill V
    for j in range(m):
        for k in range(p):
            row_v = j * p + k
            for s in range(2):
                for kp in range(p):
                    r = s * p + kp
                    if k == kp:
                        V[row_v][r] = 1
                        
    # Fill W
    for k in range(p):
        for i in range(n):
            row_w = k * n + i
            for s in range(2):
                for kp in range(p):
                    r = s * p + kp
                    if k == kp and s == 0:
                        W[row_w][r] = 1
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
