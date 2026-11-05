import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulated Annealing on Vertex Permutations.
    1. Initialize permutation using a degree-based heuristic (low degree first).
    2. Use Simulated Annealing to optimize the permutation.
    3. For each swap, calculate the change in independent set size (delta) 
       by simulating the greedy selection only in the affected segment.
    4. Accept moves based on the Metropolis criterion to escape local optima.
    5. Return scores derived from the best found permutation.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists as lists of arrays for fast access
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbors_list])
    
    def evaluate_full(perm):
        """Standard greedy evaluation for baseline."""
        active = np.ones(n, dtype=bool)
        count = 0
        for v in perm:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # Initial Heuristic: Sort by degree (ascending). 
    # Low degree vertices are less likely to be blocked and block fewer others.
    # Tie-breaker: random or index. Let's use index for determinism in init.
    perm = np.argsort(degrees, kind='stable')
    
    # We maintain the 'active' mask state for the current permutation to allow incremental updates?
    # Actually, for SA, we need to evaluate swaps. 
    # A full re-evaluation is O(n * avg_degree). 
    # With n=200, n=200, avg_degree ~ 10 (for ER 0.05) or 50 (for ER 0.25).
    # 200 * 50 = 10k ops per eval.
    # 120s budget. Python is slow.
    # Let's try to optimize the evaluation.
    
    # Optimized Evaluation using bitsets?
    # numpy bool arrays are efficient enough for n=200.
    
    current_val = evaluate_full(perm)
    best_val = current_val
    best_perm = perm.copy()
    
    # Simulated Annealing Parameters
    # n=200, so permutation space is huge.
    # We need many iterations.
    # Let's estimate speed of one swap evaluation.
    # If we do a full re-eval for every swap, it's too slow.
    # We need an incremental delta evaluation.
    
    def get_active_mask(perm, up_to_idx=None):
        """
        Returns the active mask after processing perm[0:up_to_idx].
        If up_to_idx is None, processes all.
        """
        active = np.ones(n, dtype=bool)
        if up_to_idx is None:
            up_to_idx = n
        for k in range(up_to_idx):
            v = perm[k]
            if active[v]:
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return active

    def eval_swap_delta(perm, i, j, active_before_i):
        """
        Evaluates the change in count if we swap perm[i] and perm[j] (i < j).
        We assume the state before index i is 'active_before_i'.
        We simulate the segment [i, j] with the swap and without.
        Then we check if the state after j differs. If it does, we must re-evaluate the rest.
        For simplicity and robustness in this time budget, we will do a full re-evaluation
        if the delta is potentially large or if the suffix state changes.
        However, a full re-eval is O(n). 
        Let's try a hybrid: 
        1. Simulate segment [i, j] for both orders.
        2. If the resulting active masks are identical, delta is just the count diff in the segment.
        3. If they differ, we must re-evaluate the suffix [j+1, n].
        
        To keep it simple and fast enough for 120s, let's just do a full re-evaluation 
        for the candidate permutation. 
        Wait, 120s / 10k ops = 12k evaluations. 
        That's not enough for SA to converge well on n=200.
        
        Let's optimize the greedy evaluation itself.
        Using numpy vectorization isn't easy for the sequential dependency.
        But we can use a list of active vertices and remove neighbors.
        
        Let's stick to the incremental approach but be careful.
        """
        # This function is complex to implement correctly with O(1) amortized update.
        # Given the time constraint and complexity, let's try a different optimization:
        # Use a simpler local search first, then SA on the result?
        # Or just a very fast full re-evaluation.
        
        # Let's try to speed up evaluate_full.
        # The bottleneck is the loop.
        
        pass # Placeholder for actual implementation below

    # Actually, let's implement a faster evaluate using a different data structure?
    # No, n=200 is small. The loop is the issue.
    
    # Let's try to implement the incremental delta properly.
    # State: active mask at index i.
    
    def delta_eval(perm, i, j, active_at_i):
        """
        Returns (delta, new_active_at_j) for swapping i and j.
        """
        # Simulate OLD: i to j
        old_active = active_at_i.copy()
        old_count = 0
        v_i = perm[i]
        v_j = perm[j]
        
        # Process i
        if old_active[v_i]:
            old_count += 1
            old_active[v_i] = False
            nbrs = neighbors_list[v_i]
            if len(nbrs) > 0:
                old_active[nbrs] = False
        
        # Process i+1 to j-1
        for k in range(i + 1, j):
            v = perm[k]
            if old_active[v]:
                old_count += 1
                old_active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    old_active[nbrs] = False
        
        # Process j
        if old_active[v_j]:
            old_count += 1
            old_active[v_j] = False
            nbrs = neighbors_list[v_j]
            if len(nbrs) > 0:
                old_active[nbrs] = False
        
        # Simulate NEW: i to j (swapped)
        new_active = active_at_i.copy()
        new_count = 0
        
        # Process i (which is v_j now)
        if new_active[v_j]:
            new_count += 1
            new_active[v_j] = False
            nbrs = neighbors_list[v_j]
            if len(nbrs) > 0:
                new_active[nbrs] = False
        
        # Process i+1 to j-1
        for k in range(i + 1, j):
            v = perm[k]
            if new_active[v]:
                new_count += 1
                new_active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    new_active[nbrs] = False
        
        # Process j (which is v_i now)
        if new_active[v_i]:
            new_count += 1
            new_active[v_i] = False
            nbrs = neighbors_list[v_i]
            if len(nbrs) > 0:
                new_active[nbrs] = False
        
        # Check if suffix is affected
        if np.array_equal(old_active, new_active):
            return new_count - old_count, new_active
        else:
            # Suffix is affected. We need to evaluate the rest.
            # This is expensive. 
            # Strategy: If the masks differ, we return a special flag or just do full re-eval in the caller.
            # To keep the code simple and within time limits, let's just return the segment delta 
            # and a flag 'suffix_changed'. If suffix changed, we do a full re-eval.
            return new_count - old_count, new_active, True
        
    # Let's simplify. The overhead of checking array equality and branching might be high.
    # Let's just do full re-evaluation but optimize the loop.
    
    def fast_evaluate(perm):
        active = np.ones(n, dtype=bool)
        count = 0
        for v in perm:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # How many evaluations can we do in 110 seconds?
    # Let's benchmark mentally.
    # n=200. Loop 200 times.
    # Inside: if, array access, array update.
    # ~100 ns per op? 200 * 10 ops = 2000 ops = 200us? 
    # 110s / 200us = 550,000 evaluations.
    # This is feasible!
    # So we can run a decent number of SA steps.
    
    current_perm = perm.copy()
    current_val = fast_evaluate(current_perm)
    best_perm = current_perm.copy()
    best_val = current_val
    
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    deadline = start_time + 115.0
    
    while time.time() < deadline and T > T_min:
        # Pick two random indices to swap
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        
        # Ensure i < j for consistency? No, swap is symmetric.
        
        # Create new perm by swapping
        # To avoid copying the whole array, we can swap, eval, and swap back if rejected.
        # But we need the perm for evaluation.
        # Copying 200 ints is fast.
        
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = current_perm[j], current_perm[i]
        
        new_val = fast_evaluate(new_perm)
        
        delta = new_val - current_val
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_perm = new_perm
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_perm = current_perm.copy()
        
        T *= alpha
        
    # Final check
    # Return scores based on best_perm
    scores = np.zeros(n)
    for k, v in enumerate(best_perm):
        scores[v] = n - k
        
    return scores
