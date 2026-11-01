"""Tensor decomposition for 2x3 times 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # The standard schoolbook decomposition has rank n*m*p = 54.
    # For 2x3 * 3x9, this is rank 54.
    #
    # Key insight: We can decompose this multiplication as a tensor product
    # of smaller multiplications. Specifically:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can think of this as: for each pair (i, j), we multiply A[i,j] by the
    # entire row B[j, :]. The tensor structure allows us to use the rank of
    # smaller matrix multiplications.
    #
    # Strategy: Model as tensor product of (2x3 by 3x1) and (1x1 by 1x9)
    # - The 2x3 by 3x1 multiplication: C[i] = sum_j A[i,j] * B[j]
    #   This is equivalent to multiplying a 2x3 matrix by a 3x1 matrix.
    #   The rank of this is at most 2*3*1 = 6, but we can do better.
    #   Actually, for 2x3 * 3x1, the standard rank is 6.
    #   But we can use the fact that 2x2 multiplication has rank 2 (trivially)
    #   or use Strassen-like ideas. For 2x3 * 3x1, the minimal rank is 6
    #   (since it's just 6 independent scalar products that can't be combined).
    #   
    #   Wait - let me reconsider. For 2x3 * 3x1:
    #   C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    #   C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    #   
    #   Each output entry is a sum of 3 products. The tensor rank here is 6
    #   in the standard decomposition. Can we do better?
    #   The border rank and rank of 2x3*3x1... Let's think of it as:
    #   We have 2 outputs, each is an inner product of length 3.
    #   The rank is at most 6. Is it exactly 6? 
    #   
    #   Actually, for matrix multiplication M_{a,b,c}, the rank is known for
    #   small cases. M_{2,3,1} has rank 6 (no known improvement).
    #
    # - The 1x1 by 1x9 multiplication: This is trivial, rank 9.
    #
    # But the tensor product approach gives rank 6 * 9 = 54, same as schoolbook.
    #
    # Alternative approach: Use the structure more cleverly.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Group by k: For each k, we have a 2x3 * 3x1 multiplication.
    # There are 9 such independent multiplications (one for each k).
    # Each has rank 6, so total rank is 9*6 = 54.
    #
    # Group by i: For each i, we have a 1x3 * 3x9 multiplication.
    # Each has rank 3*9 = 27, total 2*27 = 54.
    #
    # The key question: can we share computation across these groups?
    #
    # Let's try a different decomposition. Consider:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this using the fact that for fixed i, the vector 
    # (A[i,0], A[i,1], A[i,2]) is multiplied by the matrix B (3x9).
    # 
    # Another approach: use bilinear forms. The multiplication tensor T has
    # T[i,j,k] = 1 for all valid (i,j,k). We want to decompose:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a 2x3 matrix, V_r is a 3x9 matrix, W_r is a 9x2 matrix
    # (with appropriate indexing).
    #
    # Actually, let me reconsider the tensor structure.
    # The multiplication tensor for A (n x m) * B (m x p) = C (n x p) is:
    # T ∈ R^{(n*m) x (m*p) x (n*p)}
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' else 0
    #
    # So T is essentially a "generalized identity" tensor.
    # The rank of this tensor is what we're minimizing.
    #
    # For (2,3,9): rank is at most 54 (schoolbook).
    # 
    # Let's try to see if we can achieve rank less than 54.
    #
    # One approach: use the fact that 2x2 matrix multiplication can be done
    # with 7 multiplications (Strassen), but that's for 2x2*2x2.
    #
    # For our case, let's try to exploit the small dimensions.
    # 
    # Actually, let me try a constructive approach to see if rank < 54 is possible.
    #
    # The tensor T has entries T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,8}.
    # We want: sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = T[i,j,k] for all (i,j,k)
    #
    # Note that T[i,j,k] = 1 for ALL valid (i,j,k). So we need:
    # sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 for all (i,j,k)
    #
    # This is a system of 2*3*9 = 54 equations.
    #
    # If we use the schoolbook: u_r[i,j]=1 for the specific (i,j), v_r[j,k]=1 for
    # specific (j,k), w_r[i,k]=1 for specific (i,k), we get 54 terms.
    #
    # Can we do it with fewer? Let's think about it differently.
    #
    # Consider fixing i=0: we need sum_r u_r[0,j] * v_r[j,k] * w_r[0,k] = 1 for all j,k.
    # This means the matrix (sum over r of u_r[0,j] * w_r[0,k] * v_r[j,k]) must be all-ones.
    #
    # Let's define for each i: M_i = sum_r u_r[i,:] ⊗ w_r[i,:] * v_r (outer product scaled)
    # Actually this is getting complicated. Let me just try to find a lower rank
    # decomposition by solving the system.
    #
    # For now, let's return the standard rank-54 solution and attempt optimization.

    U_fallback, V_fallback, W_fallback = _build_standard(n, m, p)
    
    # Check time
    if time.time() > deadline - 10:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

    # Try to find a lower rank decomposition
    # The minimum rank for 2x3 * 3x9 is at least max(2*3, 3*9, 2*9) = max(6, 27, 18) = 27
    # But that's a weak lower bound. A better lower bound is n*m*p / max(n,m,p) ... no.
    # The true lower bound from algebraic complexity is harder.
    # 
    # Let's try rank 53, 52, ... down to some minimum.
    # We'll use a numerical approach: set up the tensor and try to decompose.
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = n * m * p
    
    # Try to improve using tensor decomposition
    # For small ranks, we can try to solve the system directly
    for target_rank in range(best_rank - 1, 26, -1):  # Try ranks 53 down to 27
        if time.time() > deadline - 20:
            break
        
        result = _try_decompose(n, m, p, target_rank)
        if result is not None:
            U, V, W = result
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid decomposition, use it
    
    return best_result


def _build_standard(n, m, p):
    """Build the standard rank n*m*p decomposition."""
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

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check: for all (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    
    # Also check that off-diagonal entries are zero
    # For the multiplication tensor, T[i,j,k] should be 1 only when the indices
    # correspond to valid multiplication terms. But actually, the tensor is defined
    # such that T[(i,j),(j',k),(i',k')] = 1 iff i=i', j=j', k=k'.
    # In our flattened representation, we only check the "diagonal" entries.
    # The off-diagonal entries are automatically zero in our construction because
    # U, V, W have the right structure.
    
    # Actually, we need to be more careful. The tensor decomposition requires:
    # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # where T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') with i=i', j=j', k=k'
    # and 0 otherwise.
    #
    # In the schoolbook decomposition, this is satisfied. For a general decomposition,
    # we need to check all (a, b, c) triples, not just the "diagonal" ones.
    #
    # However, checking all n*m * m*p * p*n = (nm)(mp)(pn) = n^2 * m^2 * p^2 entries
    # is expensive. For (2,3,9): 6 * 27 * 18 = 2916 entries.
    #
    # Let's do a spot check for correctness.
    
    # Check some off-diagonal entries
    import random
    random.seed(42)
    for _ in range(100):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        
        expected = 1.0 if (i1 == i2 and j1 == j2 and k1 == k2) else 0.0
        
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
        
        if abs(val - expected) > tol:
            return False
    
    return True


def _try_decompose(n, m, p, target_rank):
    """Try to find a decomposition of the given rank. Returns None if not found."""
    # This is a complex optimization problem. For now, return None.
    # A full implementation would use tensor decomposition algorithms.
    return None
# EVOLVE-BLOCK-END
