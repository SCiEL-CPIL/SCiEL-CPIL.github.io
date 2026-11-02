"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    Strategy:
    1. Try to find a rank-59 decomposition via local search with efficient error computation.
    2. If not found in time, return the schoolbook decomposition (rank 60).
    """
    n, m, p = DIMENSIONS
    size_u = n * m  # 12
    size_v = m * p  # 20
    size_w = p * n  # 15
    schoolbook_rank = n * m * p  # 60

    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(size_u)]
        V = [[0] * rank for _ in range(size_v)]
        W = [[0] * rank for _ in range(size_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Efficient error computation using matrix operations
    # Result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can compute: for each r, outer(U[:,r], V[:,r]) then multiply by W[:,r]
    # But more efficiently: 
    # Let AV = U @ V.T  (size_u x size_v), AV[a,b] = sum_r U[a,r]*V[b,r]
    # But we need sum_r U[a,r]*V[b,r]*W[c,r], which is not AV @ W.T
    # 
    # Alternative: stack the rank-1 tensors
    # T_approx = sum_r U[:,r,None,None] * V[:,None,r,None] * W[:,None,None,r]
    # This is a 3D tensor. We can compute it as:
    # For each (a,b): T_approx[a,b,:] = W * (U[a,:].T * V[b,:])  -- no
    # 
    # Actually: T_approx[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
    # Let P = U @ V.T, so P[a,b] = sum_r U[a,r] * V[b,r] -- this is NOT what we want
    # We want sum_r U[a,r] * V[b,r] * W[c,r], which is a 3-way sum
    #
    # We can compute this as:
    # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    # This should be efficient for small sizes.
    
    def calc_error(U, V, W):
        """Compute squared error between target T and approximation."""
        approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T - approx
        return np.dot(diff.ravel(), diff.ravel())

    def try_search(target_rank, time_limit):
        """
        Attempt to find a rank-`target_rank` decomposition using local search.
        Returns the result dict if found, else None.
        """
        start_time = time.time()
        
        # Initialize with random values on the half-integer lattice
        U = np.zeros((size_u, target_rank), dtype=np.float64)
        V = np.zeros((size_v, target_rank), dtype=np.float64)
        W = np.zeros((size_w, target_rank), dtype=np.float64)
        
        # Start with small random values
        for r in range(target_rank):
            U[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=size_u)
            V[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=size_v)
            W[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=size_w)
        
        best_error = calc_error(U, V, W)
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Temperature for simulated annealing
        T_curr = 1.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() - start_time < time_limit:
            # Pick a random variable to perturb
            mat = random.choice(['U', 'V', 'W'])
            r = random.randint(0, target_rank - 1)
            
            if mat == 'U':
                idx = random.randint(0, size_u - 1)
                old_val = U[idx, r]
                delta = random.choice([-0.5, -0.25, 0.25, 0.5, -1, 1])
                new_val = old_val + delta
                new_val = round(new_val * 4) / 4  # quarter-integer for more granularity
                U[idx, r] = new_val
            elif mat == 'V':
                idx = random.randint(0, size_v - 1)
                old_val = V[idx, r]
                delta = random.choice([-0.5, -0.25, 0.25, 0.5, -1, 1])
                new_val = old_val + delta
                new_val = round(new_val * 4) / 4
                V[idx, r] = new_val
            else:
                idx = random.randint(0, size_w - 1)
                old_val = W[idx, r]
                delta = random.choice([-0.5, -0.25, 0.25, 0.5, -1, 1])
                new_val = old_val + delta
                new_val = round(new_val * 4) / 4
                W[idx, r] = new_val
            
            new_error = calc_error(U, V, W)
            
            if new_error < best_error:
                best_error = new_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if best_error < 1e-9:
                    # Found exact solution
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
            else:
                # Simulated annealing: accept with probability exp(-delta/T)
                delta = new_error - best_error
                if delta < 1e-9:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / max(T_curr, 1e-6))
                
                if not accept:
                    # Revert
                    if mat == 'U':
                        U[idx, r] = old_val
                    elif mat == 'V':
                        V[idx, r] = old_val
                    else:
                        W[idx, r] = old_val
            
            T_curr *= cooling
            if T_curr < T_min:
                T_curr = T_min
        
        # Check if best is close enough
        if best_error < 1e-6:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
        
        return None

    # Deadline guard: leave 5s margin before the 150s kill
    deadline = time.time() + 140.0
    
    # Try rank 59 first
    remaining = deadline - time.time()
    if remaining > 10.0:
        search_time = min(remaining - 5.0, 60.0)
        result = try_search(59, search_time)
        if result:
            return result
    
    # Try rank 58 if we have time
    remaining = deadline - time.time()
    if remaining > 10.0:
        search_time = min(remaining - 5.0, 30.0)
        result = try_search(58, search_time)
        if result:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
