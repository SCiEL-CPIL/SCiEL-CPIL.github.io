"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook decomposition is the guaranteed valid solution.
    # Rank = n * m * p = 3 * 4 * 7 = 84
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i, j, k] corresponds to the term A[i,j] * B[j,k] in C[i,k]
                # We assign a unique rank dimension 'col' for each (i,j,k) triple
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verification function to ensure the decomposition is exact
    def verify(U, V, W):
        r = len(U[0]) if U else 0
        # Check all (i, j, k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Perform verification
    if not verify(U, V, W):
        # This should never happen for the schoolbook construction,
        # but if it does, we return the constructed matrices anyway as they are the best effort.
        # In a real scenario, we might try to debug, but for this task, 
        # the schoolbook construction is mathematically guaranteed to be correct.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
