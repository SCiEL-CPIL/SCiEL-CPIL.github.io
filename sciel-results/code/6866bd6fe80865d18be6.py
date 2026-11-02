"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def naive_decomposition():
        """Standard schoolbook decomposition with rank n*m*p"""
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
        return {"U": U, "V": V, "W": W}
    
    best_result = naive_decomposition()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W correctly decompose the multiplication tensor."""
        rank = U.shape[1]
        # Compute the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a better decomposition using structured patterns
    # Key idea: use the block structure and try to find rank-reducing patterns
    
    # Approach 1: Try to use a decomposition based on polynomial/FFT-like structure
    # The tensor can be viewed as a 3D array T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the (i,j) entry of first matrix and (j,k) entry of second matrix contribute to (i,k) output
    
    # Let's try a direct search for a lower-rank decomposition using linear algebra
    # We want to find the minimum rank r such that there exist U (nm x r), V (mp x r), W (pn x r)
    # with sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all valid (i,j,k)
    
    # This is equivalent to finding a tensor of rank r that equals the multiplication tensor
    # The multiplication tensor has a specific structure: T[i,j,k] = delta_{i, i} * delta_{j, j} * delta_{k, k}
    # but with the constraint that the middle index must match
    
    # Let me try a different approach: use the fact that we can combine terms
    # If we have two rank-1 terms that can be merged, we reduce rank
    
    # Try random search for a better decomposition
    import random
    random.seed(42)
    
    def try_random_decomposition(rank_target, num_attempts=100):
        """Try to find a decomposition with the given rank target using random initialization"""
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                return None
            
            U = np.random.randn(n * m, rank_target)
            V = np.random.randn(m * p, rank_target)
            W = np.random.randn(p * n, rank_target)
            
            # Scale to help with convergence
            U /= np.sqrt(rank_target)
            V /= np.sqrt(rank_target)
            W /= np.sqrt(rank_target)
            
            if verify(U, V, W, tol=1e-8):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try structured decomposition using the block structure
    # Partition: middle dim 5 = 3 + 2, output dim 6 = 3 + 3
    # This gives 4 blocks, each of which can be handled separately
    
    # For a 2x3 x 3x3 multiplication, the naive rank is 2*3*3 = 18
    # For a 2x2 x 2x3 multiplication, the naive rank is 2*2*3 = 12
    # Total naive: 18 + 18 + 12 + 12 = 60 (same as overall)
    
    # But if we can find better decompositions for each block, we can do better
    # Let's try to find a rank-7 decomposition for 2x2x2 and use it where applicable
    
    # Actually, let me try a completely different approach: use the tensor rank 
    # via explicit construction based on algebraic identities
    
    # The key identity: for matrix multiplication, we can use the "laser method"
    # or direct construction. For small cases, we can try to find the minimum rank
    # by solving a system of equations
    
    # Let's try to find a decomposition with rank 50 using a structured approach
    # based on the observation that many entries of the tensor are zero in a clever basis
    
    # Alternative: use the fact that the tensor is a "slice" of a larger structure
    # and apply known results about tensor rank
    
    # For now, let's try a heuristic: start from naive and try to zero out entries
    # while maintaining validity
    
    U = best_result["U"].copy()
    V = best_result["V"].copy()
    W = best_result["W"].copy()
    rank = best_rank
    
    # Try to find linear dependencies among the columns
    # If column r is a linear combination of other columns, we can potentially eliminate it
    # This is complex, so let's try a simpler heuristic
    
    # Try random perturbations to see if we can find a lower-rank solution
    for rank_try in range(55, 59):
        if time.time() > deadline - 20:
            break
        
        result = try_random_decomposition(rank_try, num_attempts=50)
        if result is not None:
            best_result = result
            best_rank = rank_try
            break
    
    # Try a structured approach: use the block decomposition with Strassen-like ideas
    # For the 2x2x2 subblocks, use rank 7 instead of 8
    
    # Let me try to explicitly construct a better decomposition
    # by leveraging the structure of the problem
    
    # The multiplication tensor T has T[i,j,k] = 1 if and only if the (i,j) and (j,k) 
    # entries contribute to (i,k). This is always 1 for valid indices, so T is all-ones 
    # in the appropriate subspace.
    
    # Actually, T[i,j,k] = 1 for all valid (i,j,k), which means the tensor is the 
    # "all-ones" tensor in the 10x30x12 space restricted to the multiplication pattern.
    
    # Let me try a different tactic: use the fact that we can write the product as
    # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear map, and the rank is the minimum number of scalar multiplications
    
    # Known results: for 2x2x2, rank is 7 (Strassen)
    # For our case, let's try to combine multiple Strassen blocks
    
    # Final attempt: try to find a decomposition using integer/half-integer entries
    # by solving a constrained optimization problem
    
    if time.time() < deadline - 5:
        # Try a grid search over small integer values for a reduced-rank decomposition
        for rank_try in [55, 56, 57, 58]:
            if time.time() > deadline - 5:
                break
            
            # Try to find a decomposition with this rank using a structured initialization
            U_init = np.zeros((n * m, rank_try))
            V_init = np.zeros((m * p, rank_try))
            W_init = np.zeros((p * n, rank_try))
            
            # Initialize with a pattern based on the naive decomposition
            # but try to combine some terms
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank_try:
                            break
                        U_init[i * m + j, col] = 1
                        V_init[j * p + k, col] = 1
                        W_init[k * n + i, col] = 1
                        col += 1
            
            # Now try to adjust the remaining entries to satisfy the constraints
            # This is a nonlinear problem, so use a simple iterative method
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Check current validity
                if verify(U_init, V_init, W_init, tol=1e-6):
                    best_result = {"U": U_init, "V": V_init, "W": W_init}
                    best_rank = rank_try
                    break
                
                # Try to adjust entries to reduce error
                # Compute the error for each (i,j,k)
                max_error = 0
                max_idx = None
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank_try):
                                s += U_init[i * m + j, r] * V_init[j * p + k, r] * W_init[k * n + i, r]
                            error = abs(s - 1.0)
                            if error > max_error:
                                max_error = error
                                max_idx = (i, j, k)
                
                if max_error < 1e-6:
                    break
                
                # Try to fix the worst error by adjusting a random entry
                if max_idx is not None:
                    i, j, k = max_idx
                    r = random.randint(0, rank_try - 1)
                    # Nudge one of the three entries
                    choice = random.choice(['U', 'V', 'W'])
                    if choice == 'U':
                        U_init[i * m + j, r] += random.choice([-0.1, 0.1])
                    elif choice == 'V':
                        V_init[j * p + k, r] += random.choice([-0.1, 0.1])
                    else:
                        W_init[k * n + i, r] += random.choice([-0.1, 0.1])
            
            if verify(U_init, V_init, W_init, tol=1e-6):
                best_result = {"U": U_init, "V": V_init, "W": W_init}
                best_rank = rank_try
                break
    
    # Convert to lists for the return format
    return {
        "U": best_result["U"].tolist(),
        "V": best_result["V"].tolist(),
        "W": best_result["W"].tolist()
    }
# EVOLVE-BLOCK-END
