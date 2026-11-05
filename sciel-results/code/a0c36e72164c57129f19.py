"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 21 (standard 3x3 + standard 3x4)
    # But standard 3x3 is 9, standard 3x4 is 12. Total 21.
    # This is the minimum possible if rank(3x3)=9.
    # But rank(3x3)=23.
    # So 21 is likely invalid.
    # I will use rank 35 (23 for 3x3, 12 for 3x4).
    
    rank_33 = 23
    rank_34 = 12
    total_rank = rank_33 + rank_34
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Placeholder for 23-term 3x3 decomposition
    # Using standard 9-term for 3x3 as a fallback if 23 is too complex
    # But 9 is not 23.
    # I will use standard 9-term for 3x3 and standard 12-term for 3x4.
    # Total 21.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
