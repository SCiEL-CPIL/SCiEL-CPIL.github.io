"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Partition the 3x6 B matrix into two 3x3 blocks, reducing the problem
    to two independent 2x3x3 multiplications. The rank is the sum of the ranks
    of the two sub-problems.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Strategy: Partition B into two 3x3 blocks
    # B is 3x6, split into B1 (3x3, cols 0-2) and B2 (3x3, cols 3-5)
    # C is 2x6, split into C1 (2x3, cols 0-2) and C2 (2x3, cols 3-5)
    # C1 = A * B1, C2 = A * B2
    # This is two independent 2x3x3 problems.
    
    # For each 2x3x3 problem, the schoolbook rank is 2*3*3 = 18.
    # Total rank for two such problems: 18 + 18 = 36.
    # This doesn't help directly, but let's see if we can find a lower rank for 2x3x3.
    
    # Actually, let's try to find a decomposition with rank < 36 using random search.
    # We'll try ranks from lower_bound up to best_rank - 1.
    
    found_improvement = False
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank
        max_attempts = 5000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Random integer vectors in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(dim_a, r))
            V = np.random.randint(-1, 2, size=(dim_b, r))
            W = np.random.randint(-1, 2, size=(dim_c, r))
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break

    # If no improvement found, return the schoolbook decomposition
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
