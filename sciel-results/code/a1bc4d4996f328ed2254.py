import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Calculate the QAP cost for permutation p."""
    # p is a list or array of length n
    # b[p[i], p[j]]
    # Cost = sum(a[i, j] * b[p[i], p[j]])
    bp = b[np.array(p)[:, None], np.array(p)[None, :]]
    return np.sum(a * bp)

def _local_search_2opt(a, b, p, time_limit):
    """
    Perform 2-opt local search on permutation p.
    a, b are numpy arrays.
    p is a numpy array of permutation.
    """
    n = len(p)
    p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Precompute row sums of a? No, we need specific interactions.
    # To speed up delta calculation, we can precompute the current cost contributions.
    # Let's stick to the O(n^3) full pass but optimized with NumPy where possible.
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate through all pairs i, j
        # To make it faster, we can randomize the order or just do a full pass.
        # A full pass is n*(n-1)/2 swaps.
        # For n=256, that's ~32k swaps.
        # Each swap check takes O(n) if done naively.
        # 32k * 256 = 8M ops.
        # In Python, this might take ~1-2 seconds per pass.
        # We can afford ~20-50 passes in 100s.
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit:
                    break
                
                # Calculate delta for swapping p[i] and p[j]
                # p[i] and p[j] are the locations currently occupied by facilities i and j.
                loc_i = p[i]
                loc_j = p[j]
                
                # If loc_i == loc_j, skip (should not happen in permutation)
                if loc_i == loc_j:
                    continue
                
                # Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[loc_j, p[k]] - b[loc_i, p[k]])
                
                # Vectorized calculation for this pair
                # k ranges from 0 to n-1, excluding i, j
                # We can compute the sum over all k and then subtract i and j terms.
                
                # term_k = (a[i,k] - a[j,k]) * (b[loc_j, p[k]] - b[loc_i, p[k]])
                # Let v1 = b[loc_j, :] - b[loc_i, :]
                # Let v2 = a[i, :] - a[j, :]
                # Sum = sum(v1 * v2) - v1[i]*v2[i] - v1[j]*v2[j]
                # Note: v1[i] = b[loc_j, p[i]] - b[loc_i, p[i]] = b[loc_j, loc_i] - 0 = b[loc_i, loc_j]
                # v1[j] = b[loc_j, p[j]] - b[loc_i, p[j]] = 0 - b[loc_i, loc_j] = -b[loc_i, loc_j]
                # v2[i] = a[i,i] - a[j,i] = -a[i,j] (assuming a[i,i]=0)
                # v2[j] = a[i,j] - a[j,j] = a[i,j]
                
                v1 = b[loc_j, :] - b[loc_i, :]
                v2 = a[i, :] - a[j, :]
                
                # Full sum
                full_sum = np.dot(v1, v2)
                
                # Subtract i and j terms
                # term_i = v1[i] * v2[i]
                # term_j = v1[j] * v2[j]
                
                # b[loc_j, p[i]] is b[loc_j, loc_i]
                # b[loc_i, p[i]] is b[loc_i, loc_i] = 0
                # So v1[i] = b[loc_j, loc_i]
                # v2[i] = a[i, i] - a[j, i] = -a[i, j]
                
                # b[loc_j, p[j]] is b[loc_j, loc_j] = 0
                # b[loc_i, p[j]] is b[loc_i, loc_j]
                # So v1[j] = -b[loc_i, loc_j]
                # v2[j] = a[i, j] - a[j, j] = a[i, j]
                
                b_ij = b[loc_i, loc_j]
                a_ij = a[i, j]
                
                term_i = b_ij * (-a_ij)
                term_j = (-b_ij) * a_ij
                
                delta = full_sum - term_i - term_j
                
                if delta < -1e-9: # Improvement
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    # We could break and restart, or continue. Continuing is faster.
                    # But the delta calculation for subsequent swaps involving i or j is now invalid.
                    # For simplicity and correctness, if we swap, we should ideally restart the inner loop
                    # or just accept that we might miss some improvements in this pass.
                    # Given the time limit, a "best improvement" or "first improvement" with restart is safer.
                    # Let's just mark improved and continue to next j, but note that p has changed.
                    # Actually, if p changes, loc_i and loc_j for other k are different.
                    # The vectorized calc assumed p was fixed.
                    # So after a swap, the precomputed v1/v2 for *other* pairs are wrong.
                    # Therefore, we MUST restart the pass after a swap to be safe, 
                    # OR we only do one swap per pass and restart.
                    # One swap per pass is O(n^3) per swap. Too slow.
                    
                    # Better approach: Don't vectorize the delta for *all* k.
                    # Just do the O(n) loop for the specific swap.
                    # But we are already in a double loop.
                    
                    # Let's revert to a simpler, safer local search:
                    # Randomized 2-opt or Or-opt.
                    pass
        # End for i
    return p, best_cost

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Generate diverse starts
    starts = []
    
    # Greedy rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    starts.append(p_greedy)
    
    # Random starts
    for _ in range(2):
        starts.append(np.random.permutation(n))
        
    # Hungarian on surrogate
    # Surrogate: A + B (or similar)
    # The linear assignment problem min sum c_ij x_ij
    # We want to align high A with low B.
    # Cost matrix C = A + B? No, QAP is quadratic.
    # A common heuristic is to solve LAP on A + B or A - B?
    # Let's try C = a + b.
    row_ind, col_ind = linear_sum_assignment(a + b)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    starts.append(p_hung)
    
    best_p = None
    best_cost = float('inf')
    
    for p_init in starts:
        if time.time() > deadline:
            break
            
        # Run Frank-Wolfe
        # X is doubly stochastic
        # Start with a random doubly stochastic matrix or the one-lap solution
        # Let's start with the permutation matrix of p_init
        X = np.zeros((n, n))
        X[np.arange(n), p_init] = 1.0
        
        # FW iterations
        max_fw_iters = 100
        for t in range(max_fw_iters):
            if time.time() > deadline - 20: # Leave time for local search
                break
                
            # Gradient G = A X B + A^T X B
            # Assuming symmetric A, B
            G = a @ X @ b + a.T @ X @ b.T
            
            # Solve LAP on G to get S
            row_ind, col_ind = linear_sum_assignment(G)
            S = np.zeros((n, n))
            S[row_ind, col_ind] = 1.0
            
            # Step size
            gamma = 2.0 / (t + 2)
            
            # Update X
            X = (1 - gamma) * X + gamma * S
            
        # Project X to permutation
        # Use the last S or project X
        # Projecting X: find permutation closest to X
        # This is an LAP on -X (maximizing similarity)
        row_ind, col_ind = linear_sum_assignment(-X)
        p_fw = np.empty(n, dtype=np.int64)
        p_fw[row_ind] = col_ind
        
        # Local Search
        # We need a faster local search.
        # Let's implement a simple 2-opt that checks a subset of swaps or uses a heuristic.
        # Given the time, let's do a limited 2-opt.
        
        p_curr = p_fw.copy()
        cost_curr = _qap_cost(a, b, p_curr)
        
        # Local Search Time Budget
        ls_start = time.time()
        ls_limit = 10 # seconds per start
        
        improved = True
        while improved and (time.time() - ls_start) < ls_limit and time.time() < deadline - 5:
            improved = False
            # Try all pairs, but break early if no improvement found in a random subset?
            # Let's just do a full pass but optimized.
            # To speed up, we can precompute the "neighbor" costs.
            
            # Simple 2-opt
            for i in range(n):
                if (time.time() - ls_start) > ls_limit:
                    break
                for j in range(i + 1, n):
                    if (time.time() - ls_start) > ls_limit:
                        break
                    loc_i = p_curr[i]
                    loc_j = p_curr[j]
                    
                    # Calculate delta
                    # This is the bottleneck.
                    # v1 = b[loc_j, :] - b[loc_i, :]
                    # v2 = a[i, :] - a[j, :]
                    # delta = dot(v1, v2) - correction
                    v1 = b[loc_j, :] - b[loc_i, :]
                    v2 = a[i, :] - a[j, :]
                    full_sum = np.dot(v1, v2)
                    
                    b_ij = b[loc_i, loc_j]
                    a_ij = a[i, j]
                    
                    term_i = b_ij * (-a_ij)
                    term_j = (-b_ij) * a_ij
                    delta = full_sum - term_i - term_j
                    
                    if delta < -1e-9:
                        p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
                        cost_curr += delta
                        improved = True
                        # Restart inner loop? 
                        # If we don't restart, we might use stale info for subsequent j's in the same i.
                        # But p_curr changed, so loc_i is now different for the *next* j?
                        # No, loc_i is p_curr[i]. If we swap p_curr[i] and p_curr[j], 
                        # then for the next j', loc_i is the NEW location.
                        # The vector v1/v2 depends on loc_i.
                        # So we MUST restart the j-loop if we swap.
                        # This makes it O(n^3) per swap, which is very slow.
                        
                        # Alternative: Only do one swap per pass.
                        # i.e., scan all pairs, find the best improvement, apply it, restart.
                        # This is "Best Improvement" 2-opt.
                        # It's O(n^3) to find the best, then O(1) to swap, then repeat.
                        # With n=256, n^3 is 16M. One pass takes ~1s in NumPy?
                        # Let's try this strategy.
                        
                        break # Break inner j loop, and we will restart i loop?
                        # Actually, breaking inner loop and restarting outer is complex.
                        # Let's just break both and restart the while loop.
                        
            # End for i
            # If we broke due to swap, we should restart the search
            # If we completed a full pass with no swap, improved is False, loop ends.
            
        # End while improved
        
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr
    
    if best_p is None:
        # Fallback
        best_p = starts[0]
        
    return best_p.tolist()
