"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline,
    then search for lower ranks using efficient matricization-based verification.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety

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

    # Precompute matricizations of target tensor for fast verification
    T_mat_a = T.reshape(dim_a, dim_b * dim_c)  # (6, 216)
    T_mat_b = T.reshape(dim_b, dim_a * dim_c)  # (18, 72)
    T_mat_c = T.reshape(dim_c, dim_a * dim_b)  # (12, 108)

    def verify_decomposition_fast(U, V, W):
        """
        Verify decomposition using matricizations - much faster than full tensor.
        For rank-r decomposition: T_mat_a = U @ (V ⊗ W)^T
        """
        rank = U.shape[1]
        # Build matricization of candidate: T_cand_mat_a[a, b*c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Using einsum: result[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Then reshape to (dim_a, dim_b*dim_c)
        
        # Efficient computation using einsum
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        result_mat_a = result.reshape(dim_a, dim_b * dim_c)
        result_mat_b = result.reshape(dim_b, dim_a * dim_c)
        result_mat_c = result.reshape(dim_c, dim_a * dim_b)
        
        return (np.array_equal(result_mat_a, T_mat_a) and 
                np.array_equal(result_mat_b, T_mat_b) and 
                np.array_equal(result_mat_c, T_mat_c))

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        return verify_decomposition_fast(U, V, W)

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

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Compute lower bounds from matricizations
    rank1 = np.linalg.matrix_rank(T_mat_a.astype(float))
    rank2 = np.linalg.matrix_rank(T_mat_b.astype(float))
    rank3 = np.linalg.matrix_rank(T_mat_c.astype(float))
    lower_bound = max(rank1, rank2, rank3)

    # Search for rank-32 using structured perturbation of the rank-33 solution
    # Key idea: try to merge two rank-1 components or find a dependency
    if best_rank > 32 and time.time() < deadline - 5:
        # Strategy 1: Try removing one component and adjusting others
        # This is unlikely to work directly but worth a quick check
        for remove_idx in range(33):
            if time.time() > deadline - 5:
                break
            U_try = np.delete(best_U, remove_idx, axis=1)
            V_try = np.delete(best_V, remove_idx, axis=1)
            W_try = np.delete(best_W, remove_idx, axis=1)
            if verify_decomposition(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = 32
                break

    # Strategy 2: Random search for rank 32 with small number of attempts
    if best_rank > 32 and time.time() < deadline - 3:
        np.random.seed(42)
        for attempt in range(500):
            if time.time() > deadline - 3:
                break
            # Mix of dense and sparse random
            if attempt % 3 == 0:
                U = np.random.randint(-1, 2, size=(dim_a, 32))
                V = np.random.randint(-1, 2, size=(dim_b, 32))
                W = np.random.randint(-1, 2, size=(dim_c, 32))
            elif attempt % 3 == 1:
                U = np.zeros((dim_a, 32), dtype=np.int64)
                V = np.zeros((dim_b, 32), dtype=np.int64)
                W = np.zeros((dim_c, 32), dtype=np.int64)
                for c in range(32):
                    for _ in range(np.random.randint(1, 4)):
                        U[np.random.randint(dim_a), c] = np.random.choice([-1, 1])
                    for _ in range(np.random.randint(1, 4)):
                        V[np.random.randint(dim_b), c] = np.random.choice([-1, 1])
                    for _ in range(np.random.randint(1, 4)):
                        W[np.random.randint(dim_c), c] = np.random.choice([-1, 1])
            else:
                U = np.random.randint(-2, 3, size=(dim_a, 32))
                V = np.random.randint(-2, 3, size=(dim_b, 32))
                W = np.random.randint(-2, 3, size=(dim_c, 32))
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = 32
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
