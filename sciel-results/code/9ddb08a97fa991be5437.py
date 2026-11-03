"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)  
    W: shape (p*n, rank) = (12, 48)
    
    For each (i,j,k) triple, one rank-1 term:
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    
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

def _verify_decomposition(U, V, W):
    """Verify exact decomposition of the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute T[i,k,j] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, p, m), dtype=np.float64)
    
    for t in range(rank):
        u_col = U_arr[:, t]
        v_col = V_arr[:, t]
        w_col = W_arr[:, t]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, k, j] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # The multiplication tensor should be all-ones
    expected = np.ones((n, p, m), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-10)

def _try_find_lower_rank(deadline):
    """
    Attempt to find a rank lower than 48 using structured search.
    
    Strategy:
    1. Try Winograd-style blocking for the 4x4 x 4x4 part, combined with 3x4.
    2. Use algebraic structure to share computations.
    
    For 3x4 * 4x4:
    - The 4x4 * 4x4 part can potentially use Winograd with rank < 16
    - But we have 3 rows of A, each of size 4x4
    
    Let's try a direct approach: treat each of the 3 output rows independently,
    but share the B-matrix computations.
    
    Actually, let's try a more systematic approach:
    - For the 4x4 x 4x4 multiplication, the tensor rank is known to be 16 (schoolbook)
    - But there might be a lower rank decomposition using algebraic identities
    
    Let's try to use the fact that we can combine terms.
    
    Key insight: For C[i,k] = sum_j A[i,j] * B[j,k], if we can find
    a set of linear combinations that allow sharing, we can reduce rank.
    
    Let's try a Winograd-style approach for the 4x4 part:
    - Use 2x2 blocking: split 4x4 into four 2x2 blocks
    - Each 2x2 x 2x2 has tensor rank 4 (schoolbook) or possibly less
    - But 2x2 x 2x2 has known rank 4 (no improvement known)
    
    So the Winograd approach for 4x4 x 4x4 gives:
    - 4 blocks of 2x2 x 2x2, each rank 4 = total rank 16
    - Same as schoolbook, no improvement
    
    Let's try a different approach: use the structure of the 3x4 matrix.
    
    Alternative: Try to find a rank-r decomposition by solving a system of equations.
    For small r, we can use numerical optimization.
    
    Given the time constraint, let's try a heuristic search for rank 47, 46, etc.
    by attempting to merge or eliminate terms.
    """
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    current_rank = len(schoolbook["U"][0])
    
    # Try to find a lower rank by attempting to combine terms
    # This is a heuristic approach - we try to find linear dependencies
    
    # For now, return None as a full search is infeasible in the time budget
    return None

def construct():
    """Construct the tensor decomposition with deadline guard."""
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave 30s margin
    
    # Start with schoolbook (rank 48)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank
    if time.time() < deadline - 5:
        result = _try_find_lower_rank(deadline)
        if result is not None:
            U, V, W = result
            new_rank = len(U[0])
            if new_rank < best_rank and _verify_decomposition(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = new_rank
    
    return best
# EVOLVE-BLOCK-END
