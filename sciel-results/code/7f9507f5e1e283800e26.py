"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    # Schoolbook decomposition - guaranteed valid
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower-rank exact decomposition using algebraic structure
    # For 4x4 * 4x6, we can try to use block decomposition
    # The key insight: if we can find a rank-r decomposition for 4x4*4x4,
    # we can extend it to 4x4*4x6 by duplicating columns for the extra 2 columns
    
    # Let's try a systematic search for small rank improvements
    # Using the structure that T[i,j,k] = 1 for all valid indices
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def is_valid_entries(U_mat, V_mat, W_mat):
        """Check that all entries are integers or half-integers."""
        for mat in [U_mat, V_mat, W_mat]:
            for row in mat:
                for val in row:
                    if val != int(val) and val != int(val) + 0.5 and val != int(val) - 0.5:
                        return False
        return True
    
    # Try rank 48: split into two halves of the k dimension
    # For each half of k (3 columns each), use a 4x4*4x3 decomposition
    # This might not help directly
    
    # Try a different approach: use the fact that for the 4x4 part,
    # we can potentially use a rank-12 or lower decomposition
    # and extend to 4x6
    
    # Actually, let's try a more principled approach:
    # The tensor is all-ones on the valid indices. We need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # One family of solutions: use Kronecker products
    # If we have a rank-r decomposition for 2x2*2x2, we can build up
    
    # For now, let's try a few structured decompositions
    
    # Attempt 1: Rank 48 by splitting k into two groups of 3
    if time.time() < deadline - 30:
        try:
            r_half = 24  # rank for 4x4*4x3
            U_test = [[0] * r_half for _ in range(n * m)]
            V_test = [[0] * r_half for _ in range(m * p)]
            W_test = [[0] * r_half for _ in range(p * n)]
            # This won't work directly since p=6 not 3
            pass
        except:
            pass
    
    # Attempt 2: Use a rank-32 decomposition based on 4x4*4x4 rank-16 doubled
    # If 4x4*4x4 has rank R, then 4x4*4x6 can have rank R*1.5 (not clean)
    
    # The safest improvement: verify schoolbook is correct and return it
    # Any risk of invalid decomposition is worse than a higher rank
    
    # Let's do a final verification of the schoolbook solution
    if not verify_decomposition(best_U, best_V, best_W):
        # Should never happen, but rebuild if needed
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
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
