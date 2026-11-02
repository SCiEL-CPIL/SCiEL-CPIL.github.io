"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    def time_left():
        return deadline - time.time()

    def schoolbook():
        """Return the standard rank-72 decomposition."""
        rank = n * m * p  # 72
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid decomposition of the matmul tensor."""
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # should equal 1 iff a=k*n+i, b=i*m+j, c=j*p+k for consistent (i,j,k)
        # i.e., T[k*n+i, i*m+j, j*p+k] = 1, else 0
        # Check nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = k * n + i
                    b = i * m + j
                    c = j * p + k
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check some zero entries (sample)
        rng = np.random.RandomState(42)
        for _ in range(500):
            a = rng.randint(0, p * n)
            b = rng.randint(0, n * m)
            c = rng.randint(0, m * p)
            # Check if this is a nonzero entry
            i_out = a % n
            k_out = a // n
            i_in = b // m
            j_in = b % m
            j_in2 = c // p
            k_in2 = c % p
            if i_out == i_in and k_out == k_in2 and j_in == j_in2:
                continue  # nonzero entry, skip
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
        return True

    def try_structured_rank(rank):
        """
        Try to find a rank-r decomposition using structured approach.
        
        Key insight: For (n,m,p) = (3,3,8), we can try to exploit the fact that
        the middle dimension m=3 is small.
        
        Strategy: Use the fact that for each pair (i,i'), the matrix M_{i,i'}
        (mapping from B to C) has a specific structure. We can try to share
        computation across different (i,i') pairs.
        
        Alternative: Try random search with gradient-free optimization.
        """
        if time_left() < 5:
            return None

        # Try a structured decomposition based on splitting p into blocks
        # p = 8 = 3 + 3 + 2
        # For the first 3 k-values, we can use a 3x3x3 block
        # For the next 3 k-values, another 3x3x3 block
        # For the last 2 k-values, a 3x3x2 block
        
        # 3x3x3 matmul can potentially be done in rank 23 (Strassen-like)
        # but let's start with known bounds.
        
        # Actually, let me try a different approach: random initialization + iterative refinement
        # using the fact that the system is bilinear.
        
        best_result = None
        
        for attempt in range(20):
            if time_left() < 5:
                break
                
            rng = np.random.RandomState(attempt * 12345 + 67890)
            
            # Initialize with random values
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Normalize columns to prevent blowup
            for r in range(rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Try iterative refinement: minimize ||T - sum_r u_r ⊗ v_r ⊗ w_r||^2
            # Using alternating least squares or gradient descent
            
            # For alternating approach: fix V,W, solve for U; then fix U,W, solve for V; etc.
            # But this is for linear problems. For tensor rank, we need a different approach.
            
            # Let me try a simple approach: use the constraint that for each (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # and for all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
            
            # This is a system of 5184 equations in 72*rank variables (for rank=72, that's 5184 vars).
            # For lower rank, we have fewer variables than equations, so it's overdetermined.
            
            # Let me try a completely different approach: use known results.
            # For 3x3x3 matmul, the best known rank is 23 (Strassen).
            # For 3x3x8, we can think of it as three 3x3x3 blocks plus one 3x3x2 block.
            # But the blocks aren't independent because they share the same A and B matrices.
            
            # Actually, let me think about this differently.
            # The tensor for 3x3x8 matmul has a specific sparsity pattern.
            # Let me try to use the fact that m=3 is small.
            
            # For each j (middle index), we have a "slice" of the tensor.
            # The contribution from index j is: A[:,j] * B[j,:]
            # This is an outer product of an n-vector and a p-vector.
            
            # So T = sum_{j=0}^{m-1} A[:,j] ⊗ B[j,:] ⊗ (something)
            # But this isn't quite right because A[:,j] depends on all i's.
            
            # Let me try yet another approach: use the fact that we can write
            # C = A * B = sum_j A[:,j] * B[j,:]
            # Each term A[:,j] * B[j,:] is a rank-1 matrix (outer product).
            # So C is a sum of m=3 rank-1 matrices.
            # But the rank-1 decomposition of C depends on A and B, so this doesn't
            # directly give us a bilinear decomposition of the matmul operation.
            
            # Let me try a numerical search for low-rank solutions.
            # I'll use a simple coordinate descent approach.
            
            if rank <= 72:
                # Try to find a solution by solving the nonlinear system
                # using a simple iterative method
                
                # Initialize more carefully: use the schoolbook solution as a starting point
                # and try to merge terms.
                
                U_init, V_init, W_init = schoolbook()
                
                # Try to reduce rank by merging pairs of terms
                # This is complex, so let me just try random search for now.
                
                pass
        
        return None

    # First, try to find a better rank than 72
    # Known lower bounds:
    # - The rank is at least the "subrank" which is at least max(n*m, m*p, p*n) = 24
    # - For 3x3x3, best known is 23
    # - For 3x3x8, we can try ranks from 24 to 71
    
    # Let me try a structured approach based on block decomposition
    # p = 8 = 3 + 3 + 2
    
    # For the 3x3x3 blocks, use rank 23 each (if we can find it)
    # For the 3x3x2 block, use rank 6 (schoolbook)
    # Total: 23 + 23 + 6 = 52? But this isn't quite right because the blocks
    # share the same A and B matrices.
    
    # Actually, the correct way to think about it:
    # C[:, 0:3] = A * B[:, 0:3]  (3x3 matmul)
    # C[:, 3:6] = A * B[:, 3:6]  (3x3 matmul)
    # C[:, 6:8] = A * B[:, 6:8]  (3x2 matmul)
    
    # If we use rank r1 for the first, r2 for the second, r3 for the third,
    # the total rank is r1 + r2 + r3 (since they operate on different output indices).
    
    # For 3x3x3, if we can achieve rank 23, then 23+23+6 = 52
    # For 3x3x2, schoolbook is 6, but maybe we can do better?
    # 3x3x2: at least max(9, 6, 6) = 9? No, that's not a valid lower bound.
    # Actually, for n x m x p, the rank is at least max(n*m, m*p, p*n) is NOT correct.
    # The correct lower bound is more subtle.
    
    # Let me just try to find a 3x3x3 rank-23 decomposition and use it.
    
    # Strassen's algorithm for 2x2x2 uses rank 7.
    # For 3x3x3, the best known is 23 (by Stothers, 1971, improved by others).
    
    # Let me try to implement a 3x3x3 rank-23 decomposition.
    # This is complex, so let me try a simpler approach first.
    
    # Actually, let me try a completely different strategy: use the fact that
    # we can pad the 3x3x8 problem to a 4x4x8 or similar and use known results.
    
    # Or, let me try a numerical search for the optimal rank.
    
    # Given the time constraints, let me try a few specific ranks and see if we can
    # find valid decompositions.
    
    # Try rank 60, 55, 50, 45, 40, 35, 30, 25
    for test_rank in [60, 55, 50, 45, 40, 35, 30, 25]:
        if time_left() < 10:
            break
        result = try_structured_rank(test_rank)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If we couldn't find a better rank, return schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
