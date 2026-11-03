import numpy as np
import time

N = 100
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _max_overlap_binary(h):
    """
    Calculate the maximum overlap for a binary sequence h.
    Overlap at shift k: sum_i h[i] * (1 - h[(i-k)%N])
    Since h[i] is 0 or 1, this counts the number of pairs (i, i-k) such that 
    h[i]=1 and h[i-k]=0.
    """
    max_ov = 0.0
    for k in range(N):
        # Use bit operations for speed if possible, but N=100 is small enough for numpy
        shifted = np.roll(h, k)
        # h * (1 - shifted)
        # If h[i]=1 and shifted[i]=0, product is 1.
        ov = np.sum(h * (1 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _max_overlap_continuous(h):
    """
    Calculate the maximum overlap for a continuous sequence h.
    """
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    """
    Construct the optimal sequence of step heights.
    Uses Simulated Annealing on a binary representation to find a good combinatorial structure,
    then performs a local continuous refinement.
    """
    start_time = time.time()
    
    # 1. Initialize with a random binary sequence with exactly N/2 ones.
    # This satisfies sum(h) = N/2.
    np.random.seed(42)
    
    # Start with a random permutation of N/2 ones and N/2 zeros
    h_bin = np.zeros(N, dtype=int)
    indices = np.random.choice(N, size=N//2, replace=False)
    h_bin[indices] = 1
    
    # Convert to float for potential later use, but keep binary logic for SA
    h = h_bin.astype(float)
    
    current_overlap = _max_overlap_binary(h_bin)
    best_h = h.copy()
    best_overlap = current_overlap
    best_bin = h_bin.copy()
    
    # Simulated Annealing Parameters
    # T_max needs to be large enough to allow moves that increase overlap significantly
    # Max possible overlap is N/2 (if all 1s are clustered and all 0s are clustered, 
    # actually max overlap for binary is bounded).
    # Let's estimate max possible overlap.
    # If h is [1...1, 0...0], shift 1: overlap is 0? 
    # h[i](1-h[i-k]). If h is block of 1s then 0s.
    # Let's just set T_max to a reasonable value like N/2.
    T_max = 50.0
    T_min = 1e-4
    # Number of iterations: We have ~110s. 
    # Each eval is O(N^2) ~ 10^4 ops. Very fast.
    # We can do many iterations. Let's aim for 50,000 iterations.
    num_iters = 50000
    
    # Precompute neighbors? No, just pick random i, j to swap.
    # Since sum must remain N/2, we must swap a 1 with a 0.
    
    # Get indices of 1s and 0s
    ones_indices = np.where(h_bin == 1)[0]
    zeros_indices = np.where(h_bin == 0)[0]
    
    # To allow fast swaps, we can just pick a random 1 and a random 0.
    # But we need to track the state.
    
    for i_iter in range(num_iters):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Cooling schedule: Geometric
        T = T_max * (T_min / T_max) ** (i_iter / num_iters)
        
        # Pick a random 1 to move
        idx_1 = np.random.choice(ones_indices)
        # Pick a random 0 to move
        idx_0 = np.random.choice(zeros_indices)
        
        # If they are the same (shouldn't happen) or adjacent? 
        # Just swap them.
        if idx_1 == idx_0:
            continue
            
        # Perform swap
        h_bin[idx_1] = 0
        h_bin[idx_0] = 1
        
        # Update index sets? 
        # Removing from sets is expensive. 
        # Alternative: Keep h_bin and just pick random indices from the full array, 
        # but check if one is 1 and other is 0.
        # Given N=100, checking is cheap.
        
        # Let's restart the selection strategy to be simpler and robust:
        # Pick two distinct random indices i, j.
        # If h_bin[i] != h_bin[j], swap.
        # This maintains the count.
        
        # Actually, the previous method with arrays is fine if we don't update arrays, 
        # but we might pick an index that is no longer 1 or 0.
        # So, let's just pick i, j randomly from 0..N-1.
        
        # Re-do the selection inside the loop for correctness without complex set management
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        # Undo the previous swap if we did it? No, let's stick to one strategy.
        # Let's discard the complex set tracking and use the simple i,j pick.
        # But we already swapped above. Let's undo.
        h_bin[idx_1] = 1
        h_bin[idx_0] = 0
        
        # Now do the simple swap
        if i == j:
            continue
            
        if h_bin[i] != h_bin[j]:
            # Swap
            h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
            
            new_overlap = _max_overlap_binary(h_bin)
            delta = new_overlap - current_overlap
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_overlap = new_overlap
                if new_overlap < best_overlap:
                    best_overlap = new_overlap
                    best_bin = h_bin.copy()
                    best_h = h_bin.astype(float)
            else:
                # Revert
                h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        
        # Check time periodically
        if i_iter % 1000 == 0 and time.time() - start_time > TIME_LIMIT - 5:
            break

    # 2. Local Continuous Refinement
    # The binary solution is a vertex. 
    # We can try to slightly perturb the values near 0 and 1 to see if we can lower the max overlap.
    # However, since the objective is convex, moving from a vertex into the interior 
    # generally increases the objective value for the *max* of convex functions?
    # Wait, minimizing a convex function over a convex set: the minimum is in the interior 
    # IF the function is strictly convex and the constraints allow it.
    # But here we are minimizing MAX of overlaps.
    # Let's test if a small continuous move from the best binary solution helps.
    
    # If we have time, run a quick gradient descent or SLSQP starting from the binary solution.
    if time.time() - start_time < TIME_LIMIT - 10:
        h_start = best_h.copy()
        # Ensure it's strictly inside bounds for SLSQP if needed, 
        # but SLSQP can handle bounds.
        
        # We will use SLSQP to refine.
        # Objective: _max_overlap_continuous
        # Constraints: sum = N/2, 0 <= h <= 1
        
        try:
            from scipy.optimize import minimize
            
            # Perturb slightly to avoid gradient issues at vertices?
            # SLSQP can handle vertices.
            
            def objective(x):
                return _max_overlap_continuous(x)
                
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
            bounds = [(0, 1)] * N
            
            res = minimize(objective, h_start, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                res_h = res.x
                res_score = _max_overlap_continuous(res_h)
                
                # Verify constraints
                if np.isclose(np.sum(res_h), N/2) and np.all(res_h >= 0) and np.all(res_h <= 1):
                    if res_score < best_overlap:
                        best_h = res_h
                        best_overlap = res_score
                        
        except Exception:
            pass

    # Final validation and correction
    # Ensure sum is exactly N/2
    diff = N/2 - np.sum(best_h)
    if not np.isclose(diff, 0, atol=1e-10):
        # Simple projection: subtract diff/N
        best_h = best_h - diff / N
        # Clip
        best_h = np.clip(best_h, 0, 1)
        # If clipping broke sum, revert to uniform
        if not np.isclose(np.sum(best_h), N/2, atol=1e-10):
            best_h = np.full(N, 0.5)

    return best_h
