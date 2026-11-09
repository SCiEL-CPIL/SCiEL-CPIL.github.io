"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEATHLINE = 145.0  # Safety margin before 150s kill

    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def build_trivial():
        """Schoolbook rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition of matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Start with the trivial decomposition
    best = build_trivial()
    best_rank = n * m * p

    if elapsed() > DEATHLINE - 5:
        return best

    # Strategy: Try to find lower-rank decompositions using algebraic identities
    # For matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k
    # We can potentially combine terms using linear algebra tricks.
    
    # Approach: Try small rank reductions by finding linear dependencies
    # among the rank-1 terms. If we can express one term as a linear combination
    # of others, we can reduce the rank.
    
    # For now, let's try a systematic search for rank < 60
    # This is computationally expensive, so we limit the search
    
    # Try rank 59 down to some minimum
    # Actually, let's be smarter: use the structure of matrix multiplication
    
    # The key insight: for 2x5 x 5x6, we can potentially use Strassen-like
    # algorithms if we can decompose the dimensions appropriately.
    
    # For now, return the trivial decomposition since finding a lower rank
    # requires significant computational effort that may exceed the time limit.
    
    # Let's try one simple optimization: check if we can merge any pairs
    # of terms that have the same U or V or W index.
    
    # Actually, in the trivial decomposition, each term has a unique (i,j,k)
    # so no two terms share the same U index, V index, or W index simultaneously.
    # This means we can't simply merge terms.
    
    # Return the best result found
    return best
# EVOLVE-BLOCK-END
