"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Build the target tensor and indices of non-zero entries
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                nz_indices.append((a, b, c))
    nz_indices = np.array(nz_indices)

    def schoolbook():
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U_arr, V_arr, W_arr):
        """Check if the decomposition is exact using einsum."""
        # First check non-zero entries for speed
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this for all nz_indices
        a_idx, b_idx, c_idx = nz_indices.T
        # Compute values at nz indices
        # U[a,r] * V[b,r] * W[c,r]
        # This is a batched operation
        # U[a_idx, :] has shape (N, R)
        # V[b_idx, :] has shape (N, R)
        # W[c_idx, :] has shape (N, R)
        # Sum over R
        vals = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        # Extract values at nz indices
        # Actually, computing the full tensor is O(R * N * M * P) which is R * 2500.
        # For R=50, this is 125,000 ops. Very fast.
        # For R=25, 62,500 ops.
        # So just compute the full tensor.
        if not np.allclose(vals, target, atol=1e-9):
            return False
        return True

    def try_rank(rank, max_iters=10000):
        """Try to find a rank-r decomposition via random search."""
        # Values: half-integers in [-2, 2]
        # To ensure exactness, we should use integers or half-integers.
        # Random search over a small set of values is a good strategy.
        # Let's use integers and half-integers from -2 to 2.
        # vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        vals = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            U_arr = np.random.choice(vals, size=(n * m, rank))
            V_arr = np.random.choice(vals, size=(m * p, rank))
            W_arr = np.random.choice(vals, size=(p * n, rank))
            
            if verify(U_arr, V_arr, W_arr):
                return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
        
        return None

    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = 25
    # Schoolbook: 50
    # We will try from 25 to 50.
    # To increase chances, we can use different value sets or more iterations for lower ranks.
    
    # Heuristic: Lower ranks are harder to find randomly.
    # We might need more iterations for lower ranks.
    # But time is limited.
    
    # Let's try a mixed approach:
    # 1. Try rank 50 (should be found instantly if we are lucky, but we have schoolbook)
    # 2. Try rank 49, 48...
    
    # Actually, random search for rank < 50 is very unlikely to succeed with small value sets.
    # We should try to use the structure.
    # However, for now, let's just do a faster random search with a larger value set
    # and see if we find anything. If not, return schoolbook.
    
    # One improvement: Use a larger set of values for the search.
    # And use a "greedy" approach?
    # Let's stick to random search but optimize the loop.
    
    # We can also try to fix some variables.
    # But let's just run the random search for a few seconds.
    
    # Try ranks 25 to 50.
    # Allocate more time to lower ranks?
    # Or just sweep through.
    
    # Let's try rank 50 first with random search to see if we find a non-schoolbook solution?
    # No, schoolbook is guaranteed.
    
    # Let's try ranks 25 to 49.
    # If we find one, return it.
    
    # To make it faster, we can pre-allocate arrays?
    # np.random.choice is reasonably fast.
    
    for rank in range(25, 50):
        if time.time() > deadline - 10:
            break
        
        # More iterations for lower ranks?
        # Let's use a fixed number of iterations per rank, say 1000.
        # If we have 25 ranks, that's 25,000 iterations.
        # Each iteration is ~1ms?
        # 25,000 * 1ms = 25s.
        # This fits in the budget.
        
        result = try_rank(rank, max_iters=2000)
        if result is not None:
            return result
    
    # If no improvement found, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
