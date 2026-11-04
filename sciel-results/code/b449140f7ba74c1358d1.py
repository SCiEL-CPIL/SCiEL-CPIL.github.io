"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_a = n * m   # 6
    dim_b = m * p   # 18
    dim_c = p * n   # 12

    # Build target tensor T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W):
        """Check exact decomposition via matricization (fast)."""
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Check via matricization: T_mat[a, (b,c)] = U @ (V * W.T)
        # Actually: sum_r U[a,r]*V[b,r]*W[c,r]
        # Let's verify by reconstructing: result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum for clarity and correctness
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(result, T)

    def strassen_block_decomposition():
        """
        Rank-33 decomposition: 3 blocks, each handling 2 columns of B.
        Each block = Strassen(2x2x2) for A[:,0:2] x B[0:2, 2b:2b+2]
                  + Standard(2x1x2) for A[:,2] x B[2, 2b:2b+2]
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)

        # Strassen 2x2x2 factors (7 terms)
        S_U = np.array([
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ], dtype=np.int64)
        S_V = np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ], dtype=np.int64)
        S_W = np.array([
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ], dtype=np.int64)

        # Standard 2x1x2 factors (4 terms)
        Std_U = np.array([
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ], dtype=np.int64)
        Std_V = np.array([
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ], dtype=np.int64)
        Std_W = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ], dtype=np.int64)

        col = 0
        for b in range(3):
            # Strassen part: A cols 0,1 x B rows 0,1 cols 2b,2b+1
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r, i, j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r, j, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r, kp, i]
                col += 1

            # Standard part: A col 2 x B row 2 cols 2b,2b+1
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r, i, 0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r, 0, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r, kp, i]
                col += 1

        return U, V, W

    # Start with rank-33 baseline
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33

    # Verify baseline
    if not verify(best_U, best_V, best_W):
        # Fallback: schoolbook rank 36
        best_rank = n * m * p
        best_U = np.zeros((dim_a, best_rank), dtype=np.int64)
        best_V = np.zeros((dim_b, best_rank), dtype=np.int64)
        best_W = np.zeros((dim_c, best_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, col] = 1
                    best_V[b, col] = 1
                    best_W[c, col] = 1
                    col += 1

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for lower ranks if time permits
    # Prioritize rank 32 as it is the theoretical minimum for this case (often cited)
    # and offers a significant score improvement over 33.
    found_improvement = False
    
    # Define search order: try 32 first, then 31, etc.
    # But we must respect the lower bound.
    search_ranks = [r for r in range(max(lower_bound, 32), best_rank) if r < best_rank]
    # If lower bound is higher, we just try from lower_bound up to best_rank-1
    if not search_ranks:
        search_ranks = list(range(max(lower_bound, best_rank - 5), best_rank))
        
    # Sort by rank ascending (lowest rank first)
    search_ranks.sort()

    for r in search_ranks:
        if time.time() > deadline - 15:
            break

        # Random search with structured sparsity
        # Increase attempts for the most likely target (32)
        max_attempts = 5000 if r == 32 else 2000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 15:
            attempts += 1

            # Strategy: sparse random with {-1, 0, 1} entries
            # Slightly adjust sparsity to match the "density" of known solutions
            # Rank 32 solution is likely to have similar sparsity to rank 33.
            U = np.zeros((dim_a, r), dtype=np.int64)
            V = np.zeros((dim_b, r), dtype=np.int64)
            W = np.zeros((dim_c, r), dtype=np.int64)
            for c in range(r):
                nz_u = np.random.randint(1, 4)
                nz_v = np.random.randint(1, 5)
                nz_w = np.random.randint(1, 4)
                idx_u = np.random.choice(dim_a, size=nz_u, replace=False)
                idx_v = np.random.choice(dim_b, size=nz_v, replace=False)
                idx_w = np.random.choice(dim_c, size=nz_w, replace=False)
                U[idx_u, c] = np.random.choice([-1, 1], size=nz_u)
                V[idx_v, c] = np.random.choice([-1, 1], size=nz_v)
                W[idx_w, c] = np.random.choice([-1, 1], size=nz_w)

            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
