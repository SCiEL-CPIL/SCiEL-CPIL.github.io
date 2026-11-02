"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Known lower bound: max(n*m, m*p, n*p) = max(6, 9, 6) = 9
    # But actual tensor rank might be higher
    
    # Strategy: Try to find a decomposition with rank < rank_schoolbook
    # For 2x3 * 3x3, let's try various structured approaches
    
    deadline = time.time() + 140  # safety margin
    
    # First, try the known schoolbook (guaranteed to work)
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook works
    if not _verify(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook verification failed")
        
    # The search for better decompositions is currently a placeholder.
    # We return the valid schoolbook decomposition.
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
