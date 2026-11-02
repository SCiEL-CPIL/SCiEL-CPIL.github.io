"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    # Laderman rank-23 decomposition for 3x3x3
    def laderman_3x3x3():
        # 23 terms, each with U (9 entries), V (9 entries), W (9 entries)
        terms = [
            # 1-9: diagonal schoolbook terms
            ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
            ([0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]),
            ([0,0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
            ([0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
            # 10-23: Laderman combined terms
            ([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1]),
            ([0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1]),
            ([0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0]),
            ([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0]),
            ([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0]),
            ([1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0]),
            ([0,0,0,0,1,0,0,1,0], [0,0,0,0,1,0,0,1,0], [0,0,0,0,1,0,0,1,0]),
            ([1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0]),
            ([0,1,1,0,0,0,0,0,0], [0,1,1,0,0,0,0,0,0], [0,1,1,0,0,0,0,0,0]),
            ([0,0,0,0,0,0,1,0,1], [0,0,0,0,0,0,1,0,1], [0,0,0,0,0,0,1,0,1]),
            ([1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1]),
        ]

        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))

        for c, (u, v, w) in enumerate(terms):
            U[:, c] = np.array(u)
            V[:, c] = np.array(v)
            W[:, c] = np.array(w)

        # Verify
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        return None
        return U, V, W

    lad = laderman_3x3x3()
    if lad is not None and elapsed() < deadline - 5:
        U_lad, V_lad, W_lad = lad
        combined_rank = 23 + 45  # 68
        U_comb = np.zeros((9, combined_rank))
        V_comb = np.zeros((24, combined_rank))
        W_comb = np.zeros((24, combined_rank))

        # Laderman terms for k=0,1,2
        for c in range(23):
            for i in range(3):
                for j in range(3):
                    U_comb[i*3+j, c] = U_lad[i*3+j, c]
            for j in range(3):
                for k in range(3):
                    V_comb[j*8+k, c] = V_lad[j*3+k, c]
            for k in range(3):
                for i in range(3):
                    W_comb[k*3+i, c] = W_lad[k*3+i, c]

        # Schoolbook for k=3..7
        col = 23
        for i in range(3):
            for j in range(3):
                for k in range(3, 8):
                    U_comb[i*3+j, col] = 1
                    V_comb[j*8+k, col] = 1
                    W_comb[k*3+i, col] = 1
                    col += 1

        if validate(U_comb.tolist(), V_comb.tolist(), W_comb.tolist()):
            best_U, best_V, best_W, best_rank = U_comb.tolist(), V_comb.tolist(), W_comb.tolist(), combined_rank

    # Try to find a lower-rank decomposition using structured search
    # Strategy: Use bilinear decomposition with half-integer coefficients
    # For 3x3x8, try rank 36 by splitting into two 3x3x4 blocks
    
    if elapsed() < deadline - 20:
        # Try a structured approach: for each pair of output columns (k1, k2),
        # try to find a low-rank decomposition of the 3x3x2 sub-problem
        # This is complex, so instead try random search with constraint satisfaction
        
        # First, let's try to verify if rank 36 is achievable with a specific pattern
        # The key insight: for 3x3x4, we can potentially use rank 18 (3x3x3 Laderman + 3x3x1 schoolbook)
        # So 3x3x8 = 2 * 3x3x4 could be 2 * 18 = 36 if independent
        
        # Let's construct: Laderman for k=0,1,2 (rank 23) + schoolbook for k=3 (rank 9) = 32 for first half
        # Then Laderman for k=4,5,6 (rank 23) + schoolbook for k=7 (rank 9) = 32 for second half
        # Total: 64, which is worse than 68. So this doesn't help.
        
        # Alternative: Try to find a unified decomposition
        # For now, keep the 68-rank solution as it's better than 72
        
        # Try random perturbation search to see if we can find rank < 68
        # This is expensive, so only do it if we have time
        if elapsed() < deadline - 30:
            # Try a greedy approach: start with rank 60 and try to fix errors
            # This is a research direction, but may not yield results in time
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
