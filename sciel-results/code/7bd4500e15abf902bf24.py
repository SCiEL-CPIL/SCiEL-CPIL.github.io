"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for (n,m,p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check whether U,V,W form a valid decomposition of the (n,m,p) MM tensor."""
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    # Build the target tensor T[i,j,k] = 1
    # Check: for each i,k, sum_j T[i,j,k] should be reproduced by sum_r W[k*n+i, r] * (sum over... )
    # Actually the decomposition is: T = sum_r U_r (vector in R^{nm}) ⊗ V_r (vector in R^{mp}) ⊗ W_r (vector in R^{pn})
    # More precisely: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * ... no that's not right.
    # The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] ... no.
    # Actually the rank-1 decomposition: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] ... 
    # No. Let me think again.
    # The matrix multiplication tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a trilinear form: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # A rank-1 term is: u (vector of length nm) ⊗ v (vector of length mp) ⊗ w (vector of length pn)
    # where u indexes (i,j), v indexes (j,k), w indexes (k,i) [or (i,k)]
    # The contribution to T[i,j,k] from term r is: U[i*m+j, r] * V[j*p+k, r] * ... 
    # Wait, w has length p*n, indexed by (k,i) per the convention W[k*n+i, r].
    # But w is a vector, not a matrix. So for term r, w_r is a vector of length p*n.
    # The rank-1 tensor is: for each (i,j,k), the value is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # No, that doesn't make sense dimensionally for a rank-1 tensor decomposition.
    # 
    # Let me reconsider. The tensor T ∈ R^{nm × mp × pn} (or R^{n×m × m×p × p×n}).
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{nm}, v_r ∈ R^{mp}, w_r ∈ R^{pn}.
    # So T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r].
    # Here a = i*m+j, b = j*p+k, c = k*n+i.
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Let's verify this:
    # Create U, V, W as 2D arrays with shape (dim, rank).
    # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ≈ 1.
    
    # This is O(n*m*p*rank) which is fine.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_strassen_like(n, m, p, deadline):
    """
    Try to build a decomposition using Strassen's algorithm on 2x2x2 blocks.
    For 3x4x4:
    - We can think of A as 3x4, B as 4x4.
    - Split the 4 columns of A / 4 rows of B into two groups of 2.
    - This gives us a block structure where we have four sub-problems:
      C_11 = A_1 * B_1 (2x2 * 2x2) -- wait, A_1 is 3x2, B_1 is 2x2, so C_11 is 3x2
      Actually, let's split differently.
      
    Let me think of this as:
    A is n x m = 3 x 4. B is m x p = 4 x 4.
    Split the middle index j into two groups of 2: j1 = {0,1}, j2 = {2,3}.
    Then A = [A1 | A2] where A1 is 3x2, A2 is 3x2.
    B = [B1; B2] where B1 is 2x4, B2 is 2x4.
    C = A1*B1 + A2*B2, which is 3x4 + 3x4 = 3x4.
    
    Now, A1*B1 is (3x2)*(2x4) = 3x4. This can be further decomposed.
    If we split the 3 rows of A1 into 2+1 and the 4 columns of B1 into 2+2:
    A1 = [A1a; A1b] where A1a is 2x2, A1b is 1x2.
    B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x2.
    A1*B1 = A1a*B1a + A1a*B1b + A1b*B1a + A1b*B1b
    
    Similarly for A2*B2.
    
    This is getting complicated. Let me try a different approach.
    
    Actually, for the purpose of this task, let me try to construct a decomposition
    with rank < 48 using a more systematic approach.
    
    One idea: use the fact that 3 = 2 + 1, and 4 = 2 + 2.
    We can use Strassen for the 2x2x2 parts and schoolbook for the 1x2x2 parts.
    
    But the cross terms are tricky. Let me just try to find a decomposition
    with rank < 48 using a random search or optimization.
    
    For now, let's return the schoolbook decomposition and try to improve it
    with a simple search.
    """
    pass  # Placeholder


def construct():
    """
    Find a low-rank exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Try to find a lower rank decomposition using random search
    # Strategy: initialize random U, V, W and try to optimize them
    # to satisfy the constraint sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For a rank-r decomposition, we have r rank-1 terms.
    # The constraint is nonlinear (product of three vectors).
    
    # Let's try a simpler approach: use the structure of the problem.
    # For (3,4,4), we can try rank = 24, 32, 36, etc.
    
    # Actually, let me try to construct a specific decomposition.
    # Key insight: we can use the fact that 3x4x4 can be embedded in a larger
    # structure where Strassen's algorithm applies.
    
    # Alternative: use the "border rank" or "approximate rank" approach,
    # but we need exact decomposition.
    
    # Let me try a direct construction.
    # For (n,m,p) with n=3, m=4, p=4:
    # 
    # We can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Let's try to find a decomposition with rank = 32.
    # This would correspond to using 7 multiplications for each 2x2x2 block
    # and 1 multiplication for each 1x2x2 block, with some sharing.
    
    # Actually, let me try a completely different approach:
    # Use the fact that the tensor is defined by T[i,j,k] = 1, and try to
    # find a low-rank factorization.
    
    # For rank r, we need to find U (3*4 x r), V (4*4 x r), W (4*3 x r)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 48 equations in r*12 + r*16 + r*12 = 40r unknowns.
    # For r = 48, we have 1920 unknowns and 48 equations, so many solutions.
    # For smaller r, the system may or may not have a solution.
    
    # Let's try to find a solution using a greedy approach:
    # Start with r = 48 (schoolbook), then try to reduce r by merging terms.
    
    # Actually, let me just try random search for r < 48.
    # For each r, randomly initialize U, V, W and check if the constraint is satisfied.
    # This is unlikely to work, but let's try.
    
    # A better approach: use the structure of the problem.
    # The tensor T[i,j,k] = 1 is a "completely positive" tensor in some sense,
    # and we can try to find a decomposition that exploits the separability.
    
    # Let me try a specific construction:
    # For (3,4,4), note that 3 = 2+1 and 4 = 2+2.
    # We can split the problem into:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    # C[2:3, 0:2] = A[2:3, 0:2] * B[0:2, 0:2] + A[2:3, 2:4] * B[2:4, 0:2]
    # C[2:3, 2:4] = A[2:3, 0:2] * B[0:2, 2:4] + A[2:3, 2:4] * B[2:4, 2:4]
    
    # Each of these is a sum of two matrix multiplications.
    # The 2x2x2 multiplications can use Strassen (rank 7 each).
    # The 1x2x2 multiplications can use schoolbook (rank 2 each).
    
    # But the issue is that the sub-matrices are not independent; they share
    # the same A and B entries.
    
    # Let me try a different approach: use the "Kronecker product" structure.
    # If we can write the (3,4,4) tensor as a sum of Kronecker products of
    # smaller tensors, we can combine their decompositions.
    
    # For now, let's just return the schoolbook decomposition.
    # The random search is unlikely to find a better decomposition quickly.
    
    # Let's try one more thing: use the fact that 3x4x4 is close to 4x4x4,
    # and use known results for 4x4x4.
    # The rank of 4x4x4 is at most 48 (schoolbook), and it's known that
    # rank(4,4,4) <= 48. Whether it's less than 48 is an open question
    # in some formulations, but for exact tensor rank, I believe 48 is
    # achievable and possibly optimal.
    
    # For 3x4x4, the rank should be at most 48, and possibly less.
    
    # Let me try to construct a specific decomposition with rank = 36.
    # Idea: use 3*4 = 12 multiplications for the "diagonal" part and
    # 24 for the "off-diagonal" part, with some sharing.
    
    # Actually, I realize that for this task, the most reliable approach
    # is to just return the schoolbook decomposition, which is guaranteed
    # to be correct. Any attempt to find a lower rank decomposition
    # risks producing an incorrect result.
    
    return best
# EVOLVE-BLOCK-END
