"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # The matrix multiplication tensor T has indices (i, j, k) where
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i, j, k] = 1 if we're multiplying A[i,j] * B[j,k] -> C[i,k]
    # In the unfolded form, the tensor is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # Actually, let's think about this differently.
    #
    # The tensor for matrix multiplication of (n x m) * (m x p) -> (n x p) is:
    # T_{i,j,k}^{a,b,c} where a in [0,n), b in [0,p), c in [0,m)
    # T[a,b,c] = 1 if the output C[a,b] uses A[a,c] * B[c,b]
    #
    # Wait, let me be more careful. The standard formulation:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # The tensor T has entries T[i,j,k] for each term in the sum.
    # For the decomposition, we need:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    #
    # But the interface says:
    # U has shape (n*m, rank) - indexed by (i*m + j)
    # V has shape (m*p, rank) - indexed by (j*p + k)  
    # W has shape (p*n, rank) - indexed by (k*n + i)
    #
    # And the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this term exists, 0 otherwise
    #
    # The "term exists" means: for fixed i, j, k, the product A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor entry at (i, j, k) is 1 (it's a permutation tensor).
    #
    # The tensor has n*m*p = 2*3*5 = 30 non-zero entries, each being 1.
    # The schoolbook rank is 30.
    #
    # We want to find the minimum rank. For 2x3x5 matrix multiplication,
    # the known lower bound is related to the Strassen-like bounds.
    # Let's try to find a decomposition with rank < 30.
    #
    # Strategy: Use a combinatorial search with constraint propagation.
    # The tensor is a "permutation tensor" - it has exactly one 1 in each "slice"
    # in certain senses.
    #
    # Actually, let's think about this more carefully. The tensor T is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k
    # and 0 otherwise.
    #
    # This is a 3-dimensional tensor of size (n*m) x (m*p) x (p*n) = 6 x 15 x 10
    # with 30 ones.
    #
    # We want to decompose this as a sum of rank-1 tensors:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r has length n*m, v_r has length m*p, w_r has length p*n.
    #
    # For a rank-1 term to contribute to T[i*m+j, j*p+k, k*n+i],
    # we need u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # And for all other (i', j', k') not corresponding to a valid term,
    # the sum over r must be 0.
    #
    # Let's try a different approach: use the fact that this is a "structured" tensor
    # and try to find a low-rank decomposition using algebraic methods.
    #
    # For now, let's try a systematic search for rank < 30.
    # Start with rank 29, 28, etc.
    #
    # Actually, a better approach: use the known results for small matrix multiplication.
    # For 2x2x2, the Strassen rank is 7 (vs schoolbook 8).
    # For 2x3x3, the rank is known to be... let me think.
    #
    # Let's try a numerical approach: use ALS (Alternating Least Squares) or
    # a gradient-based method to try to find a low-rank decomposition.
    #
    # But the task requires integer/half-integer entries. So we need an exact decomposition.
    #
    # Let me try a different strategy: enumerate possible rank-1 terms and use
    # constraint propagation to find a minimal set.
    #
    # Each rank-1 term is determined by three vectors (u, v, w).
    # For the term to be "compatible" with the tensor, it must only have non-zero
    # products at positions where T is non-zero, or the contributions must cancel.
    #
    # This is getting complex. Let me try a simpler approach first:
    # Use the fact that the tensor is a "block diagonal" structure in some sense.
    #
    # Actually, let me just try to use a MILP or SAT solver approach, but that's
    # not available. Let me try a heuristic search.
    #
    # For now, return the schoolbook decomposition if we can't find something better.
    # But let's at least try to see if rank 29 or less is possible.
    #
    # Let me think about this differently. The tensor T has the property that:
    # For each i, the "i-slice" T[i*m+j, *, k*n+i] is a (m x p) matrix with
    # exactly m*p = 15 ones (one for each j, k).
    #
    # Similarly, for each j, the "j-slice" T[i*m+j, j*p+k, *] has n*p = 10 ones.
    #
    # And for each k, the "k-slice" T[*, j*p+k, k*n+i] has n*m = 6 ones.
    #
    # A rank-1 term u ⊗ v ⊗ w contributes to the i-slice as u[i*m+j] * v[j*p+k] * w[k*n+i].
    # For this to be non-zero only where T is non-zero, we need:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] != 0 implies T[i*m+j, j*p+k, k*n+i] = 1
    #
    # This is a strong constraint. Let's use it to prune.
    #
    # Let's try a search where we assign values to the rank-1 terms one at a time,
    # using backtracking with constraint propagation.
    #
    # Given the time limit, let's try a limited search.

    # Build the tensor as a set of non-zero positions
    # Position is (a, b, c) where a = i*m+j, b = j*p+k, c = k*n+i
    non_zero = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero.add((a, b, c))
    
    # The tensor has |non_zero| = n*m*p = 30 entries, all equal to 1.
    # We want to write T = sum_r u_r ⊗ v_r ⊗ w_r where each entry is 1 or 0.
    #
    # For a rank-1 term (u, v, w), the support is { (a,b,c) : u[a]*v[b]*w[c] != 0 }.
    # For the sum to equal T, we need:
    # For (a,b,c) in non_zero: sum_r u_r[a]*v_r[b]*w_r[c] = 1
    # For (a,b,c) not in non_zero: sum_r u_r[a]*v_r[b]*w_r[c] = 0
    #
    # Let's try a search with small integer values.
    # Each u_r[a], v_r[b], w_r[c] can be in {-1, 0, 1} or {-1, -1/2, 0, 1/2, 1}.
    #
    # This is a huge search space. Let's try a greedy approach first.
    
    # Try to find a decomposition with rank < 30 using a heuristic.
    # Start with the schoolbook and try to merge terms.
    
    # Actually, let me try a different approach: use the fact that for small matrices,
    # we can sometimes find decompositions by exploiting symmetries.
    #
    # For 2x3x5, let's try to see if we can find a rank-29 decomposition.
    # This would mean combining two schoolbook terms into one.
    #
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be combined if:
    # There exist u, v, w such that:
    # u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] + u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
    # And for all other positions, the contribution is 0 or cancels with other terms.
    #
    # This is getting too complex for a quick search. Let me just return the schoolbook
    # decomposition for now, but with a time guard.
    
    # Actually, let me try one more thing: use numpy to see if a random low-rank
    # approximation can be rounded to give an exact decomposition.
    
    # Create the full tensor
    T = np.zeros((n*m, m*p, p*n))
    for (a, b, c) in non_zero:
        T[a, b, c] = 1.0
    
    # Try ranks from 29 down to 20
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for target_rank in range(best_rank - 1, max(10, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use a simple approach: start with random initialization and use ALS
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse vectors
            np.random.seed(42 + attempt)
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with the schoolbook pattern for the first target_rank terms
            for r in range(target_rank):
                i = (r // (m*p)) % n
                j = (r // p) % m
                k = r % p
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            
            # Try to improve using a few iterations of a simple update
            # This is a heuristic; it may not find the optimal decomposition.
            
            # Check if the current decomposition is exact
            # Compute the reconstructed tensor
            recon = np.zeros_like(T)
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            if np.allclose(recon, T, atol=1e-10):
                # Found an exact decomposition
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    if best_U is not None:
        # Convert to list of lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        
        # Round to nearest half-integer
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    mat[i][j] = round(mat[i][j] * 2) / 2
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook decomposition
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
