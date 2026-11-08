"""Matrix multiplication 4x4 x 4x5 tensor decomposition.

Strategy: The task asks for a rank-minimal exact tensor decomposition of the
4x4 x 4x5 matrix multiplication. We construct the schoolbook decomposition of
rank n*m*p = 80 and then attempt to reduce the rank by finding linear
dependencies among the rank-1 terms via algebraic methods.

Known results: The matrix multiplication tensor M_{4,4,5} has a known rank.
For M_{n,n,p} with small dimensions, the rank is known to be n*m*p in many
cases, but sometimes can be reduced. For 4x4x5, we attempt to find a
decomposition with rank < 80 using a systematic search.

Approach:
1. Start with the schoolbook decomposition (rank 80).
2. Try to find a lower-rank decomposition by exploiting the structure.
3. Use the fact that for matrix multiplication, the tensor has a specific
   combinatorial structure that may allow rank reduction.

Given the 120s time limit, we focus on:
- A direct algebraic construction if one is known
- Otherwise, return the schoolbook decomposition

The known rank of M_{4,4,4} is 21 (Strassen-like), but for M_{4,4,5} the
rank is not as well-studied. We'll try to find a decomposition with rank
less than 80 using a greedy or algebraic approach.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 80
    
    # First, build the schoolbook decomposition as a fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[mat[i, j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    fallback_result = {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    
    # Check if we've exceeded the deadline
    if time.time() > DEADLINE:
        return fallback_result
    
    # Try to find a lower-rank decomposition
    # The key insight: for matrix multiplication M_{n,m,p}, the tensor is:
    # T[(i,j), (j,k), (i,k)] = 1 for all i,j,k, and 0 otherwise.
    # A rank-1 decomposition term r contributes U[a,r]*V[b,r]*W[c,r] to T[a,b,c].
    
    # We can try to find a set of rank-1 tensors whose sum equals T.
    # Each rank-1 tensor is specified by three vectors (u, v, w) of appropriate
    # lengths, and contributes u⊗v⊗w.
    
    # Strategy: Try to find a decomposition with rank < 80 by solving a
    # system of bilinear equations. This is generally hard, but for small
    # cases we can try structured approaches.
    
    # One approach: Use the fact that if we can find a rank-r decomposition
    # where each term has a specific support pattern, we can reduce the rank.
    
    # For M_{4,4,5}, let's try a systematic search over possible rank-1
    # tensors with small integer entries.
    
    # A rank-1 tensor u⊗v⊗w is determined by:
    # - u: vector of length n*m = 16
    # - v: vector of length m*p = 20
    # - w: vector of length p*n = 20
    # The tensor entry T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    
    # The constraint is: for each (a,b,c), the sum over r must equal the
    # target tensor value (1 if (a,b,c) corresponds to a valid (i,j,k) triple,
    # 0 otherwise).
    
    # There are 16*20*20 = 6400 tensor entries, but only 16*4*5 = 320 are
    # non-zero (all equal to 1). The rest must be 0.
    
    # This is a very hard optimization problem in general. Let's try a
    # different approach: exploit known algebraic structures.
    
    # For matrix multiplication, one known technique is to use the "laser
    # method" or to exploit the group structure. However, for such small
    # dimensions, the schoolbook decomposition may already be optimal or
    # close to optimal.
    
    # Let's try a specific construction: if we can find a rank-r decomposition
    # where r < 80, we need to solve a system of bilinear equations.
    
    # Alternative approach: Try to find a decomposition using the structure
    # of the multiplication table. For M_{4,4,5}, we can try to merge
    # some of the schoolbook terms.
    
    # Let's try a greedy approach: start with the schoolbook decomposition
    # and try to find linear dependencies that allow us to reduce the rank.
    
    # Actually, a better approach for small cases: enumerate possible rank-1
    # tensors with small integer/half-integer entries and try to find a
    # combination that sums to the target tensor.
    
    # Given the time constraint, let's try a more targeted approach.
    
    # For M_{n,n,p} with n=4, p=5, one can try to use the following
    # construction based on the fact that 4 = 2*2:
    
    # Use the Strassen-like decomposition for the 2x2 block structure.
    # Divide the 4x4 matrix into 2x2 blocks of 2x2 matrices, and similarly
    # for the other factors.
    
    # This gives a block-structured problem that might be solvable with
    # fewer multiplications.
    
    # However, implementing a full Strassen-like decomposition for 4x4x5
    # is complex. Let's instead try to verify if the schoolbook rank is
    # already minimal by checking known lower bounds.
    
    # Known lower bound for M_{n,n,p} rank: at least max(n*m, m*p, n*p) =
    # max(16, 20, 20) = 20. But this is a very weak bound.
    
    # A better lower bound: the rank of M_{n,m,p} is at least the border rank,
    # which is at least the geometric rank. For matrix multiplication, the
    # border rank equals the rank, and there are stronger lower bounds.
    
    # Given the time constraints, let's try a computational approach:
    # Use numerical optimization to find a lower-rank decomposition.
    
    if time.time() > DEADLINE:
        return fallback_result
    
    # Try to find a rank-r decomposition for r < 80 using a random
    # initialization and gradient descent on the bilinear equations.
    
    # The loss function: sum over all (a,b,c) of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
    
    # This is a non-convex optimization problem, but for small r it might
    # find a solution.
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from a small value up to the schoolbook rank
    # Start with a reasonable upper bound and try to go lower
    
    # For M_{4,4,5}, let's try ranks from 20 up to 79
    # But this is computationally expensive. Let's try a few specific ranks.
    
    def try_rank_decomposition(r, num_iters=500, lr=0.01, num_restarts=5):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > DEADLINE:
            return None
        
        best_loss = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            for it in range(num_iters):
                if time.time() > DEADLINE:
                    break
                
                # Compute the tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Efficient computation using einsum
                T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                
                # Compute loss
                diff = T_comp - T_target
                loss = np.sum(diff * diff)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = (U.copy(), V.copy(), W.copy())
                
                # If loss is very small, we found a solution
                if loss < 1e-10:
                    return (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                # dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                # dL/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
                # dL/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                
                dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                # Gradient descent update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Periodic increase in learning rate if stuck
                if it % 100 == 99 and loss > 1e-6:
                    lr = min(lr * 2, 0.1)
        
        if best_params is not None and best_loss < 1e-8:
            return best_params
        return None
    
    # Try to find a decomposition with rank < 80
    # Start with the schoolbook rank and try to go lower
    found_decomp = None
    
    # Try a few specific ranks that might work
    # For M_{4,4,4}, rank 21 is known. For M_{4,4,5}, let's try various ranks.
    
    # Actually, let's first try to see if we can get below 80 with a
    # reasonable search.
    
    # Try rank 60, 50, 40, 30, 25, 20
    for r in [60, 50, 40, 30, 25, 20]:
        if time.time() > DEADLINE:
            break
        
        result = try_rank_decomposition(r, num_iters=300, lr=0.05, num_restarts=3)
        if result is not None:
            found_decomp = (r, result)
            break
    
    if found_decomp is not None:
        r, (U, V, W) = found_decomp
        # Convert to lists
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # If no lower-rank decomposition found, return the schoolbook one
    return fallback_result
# EVOLVE-BLOCK-END
