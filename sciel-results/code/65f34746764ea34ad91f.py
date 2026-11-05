"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    
    def build_decomp_base(n, m):
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    def build_decomp_full(n, m, p):
        if p == 1:
            return build_decomp_base(n, m)
        
        U1, V1, W1 = build_decomp_full(n, m, 1)
        U2, V2, W2 = build_decomp_full(n, m, p - 1)
        
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for r in range(rank1):
            for i in range(n * m):
                U[i][r] = U1[i][r]
            for j in range(m):
                V[j * p + (p - 1)][r] = V1[j][r]
            for i in range(n):
                W[(p - 1) * n + i][r] = W1[i][r]
                
        offset = rank1
        for r in range(rank2):
            for i in range(n * m):
                U[i][offset + r] = U2[i][r]
            for j in range(m):
                for k in range(p - 1):
                    V[j * p + k][offset + r] = V2[j * (p - 1) + k][r]
            for k in range(p - 1):
                for i in range(n):
                    W[k * n + i][offset + r] = W2[k * n + i][r]
                    
        return U, V, W

    U, V, W = build_decomp_full(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
