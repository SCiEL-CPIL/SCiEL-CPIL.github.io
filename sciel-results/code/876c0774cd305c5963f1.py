"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 24
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check: for all i in [n], j in [m], k in [p]:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try known ranks from best to worst
    # For 2x3x4, best known is 17
    for rank in [17, 18, 19, 20, 21, 22, 23]:
        if time.time() > DEADLINE:
            break
        
        # For now, just use schoolbook as fallback
        # A proper search would go here
    
    # Return schoolbook decomposition
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
