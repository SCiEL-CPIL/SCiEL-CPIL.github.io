"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    import numpy as np
    
    # Heuristic search for low rank decomposition
    # Target rank 24 for 2x3x7
    
    def check_decomposition(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for c in range(rank):
                    M += np.outer(U_r[i, :, c], V_r[:, k, c]) * W_r[k, i, c]
                if not np.allclose(M, np.eye(m)):
                    return False
        return True

    # Try to find a rank 24 solution
    rank_target = 24
    np.random.seed(42)
    
    for attempt in range(5000):
        U = np.random.randint(-2, 3, (n*m, rank_target))
        V = np.random.randint(-2, 3, (m*p, rank_target))
        W = np.random.randint(-2, 3, (p*n, rank_target))
        
        if check_decomposition(U, V, W, rank_target):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
        # Perturb
        if attempt % 10 == 0:
            idx = np.random.randint(n*m)
            col = np.random.randint(rank_target)
            U[idx, col] += np.random.randint(-1, 2)
            
            idx = np.random.randint(m*p)
            col = np.random.randint(rank_target)
            V[idx, col] += np.random.randint(-1, 2)
            
            idx = np.random.randint(p*n)
            col = np.random.randint(rank_target)
            W[idx, col] += np.random.randint(-1, 2)

    # Fallback to schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
