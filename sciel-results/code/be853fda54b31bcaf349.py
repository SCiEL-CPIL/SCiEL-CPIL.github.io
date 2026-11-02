"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 130.0
    
    # Schoolbook decomposition as fallback (rank 36)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U, V, W):
        """Verify exact decomposition using integer arithmetic."""
        U2 = np.array([[int(round(x * 2)) for x in row] for row in U], dtype=np.int64)
        V2 = np.array([[int(round(x * 2)) for x in row] for row in V], dtype=np.int64)
        W2 = np.array([[int(round(x * 2)) for x in row] for row in W], dtype=np.int64)
        rank = U2.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U2[i*m+j, c] * V2[j*p+k, c] * W2[k*n+i, c]
                    if s != 8:
                        return False
        return True
    
    # Return schoolbook - guaranteed correct
    return schoolbook()
# EVOLVE-BLOCK-END
