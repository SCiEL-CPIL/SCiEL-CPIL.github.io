"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    # We need to find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    # The tensor T has indices (i, j, k) where i in 0..1, j in 0..2, k in 0..3.
    # T[i,j,k] = 1.
    # We want to decompose T as sum_{r=1}^{R} U[i,j,r] V[j,k,r] W[k,i,r].
    # The output factor follows W[k*n+i, r].
    #
    # The schoolbook decomposition has rank 2*3*4 = 24.
    # We can try to find a lower rank decomposition.
    #
    # Let's use a random search approach with integer/half-integer constraints.
    # We'll try ranks from a lower bound up to 24, and for each rank, do a random search.
    #
    # The lower bound for rank of matrix multiplication tensor is at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12.
    
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115 seconds, leave 5 seconds margin
    
    import random
    
    def verify(U, V, W):
        # Verify the decomposition
        n, m, p = DIMENSIONS
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def random_search(rank, max_iters=1000):
        # Random search for a decomposition of given rank
        n, m, p = DIMENSIONS
        for it in range(max_iters):
            # Generate random U, V, W with integer/half-integer entries
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                return U, V, W
        return None
    
    # Try ranks from 12 to 24
    for rank in range(12, 25):
        if time.time() > deadline:
            break
        
        result = random_search(rank, max_iters=200)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If no better decomposition found, return schoolbook
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
# EVOLVE-BLOCK-END

