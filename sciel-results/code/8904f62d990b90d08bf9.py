"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    rank = m * p  # 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, c] = 1 if (i*m + j) % m == c // m
    # Note: m=5. c = 5*j_c + r_c. c // 5 = j_c.
    # Row u = 5*i + j. u % 5 = j.
    # Condition: j == j_c.
    for u in range(n * m):
        j = u % m
        for c in range(rank):
            j_c = c // m
            if j == j_c:
                U[u][c] = 1
                
    # V[j*p + k, c] = 1 if j*p + k == c
    # V is Identity matrix of size m*p x m*p
    for v in range(m * p):
        V[v][v] = 1
        
    # W[k*n + i, c] = 1 if c % p == k // ? 
    # Wait, my derivation was:
    # w = k*n + i. n=2. w = 2k + i.
    # k = w // n.
    # Condition: c % p == k. (p=5).
    # c = 5*j_c + r_c. c % 5 = r_c.
    # So condition: r_c == k.
    for w in range(p * n):
        k = w // n
        for c in range(rank):
            r_c = c % p
            if r_c == k:
                W[w][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
