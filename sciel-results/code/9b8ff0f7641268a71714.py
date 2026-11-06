import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build dense incidence matrix M of shape (n, Np)
    # M[i, pid] = 1 if position i is in progression pid, else 0
    M = np.zeros((n, Np), dtype=np.int32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1
    
    def compute_sums(signs):
        # signs is (n,) vector of -1, 1
        # sums is (Np,) vector of signed sums
        # sums = signs @ M
        return signs @ M
    
    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = float('inf')
    
    # Multi-restart Batch Flip Local Search
    # Instead of single flips, we consider flipping a subset of signs.
    # A powerful heuristic: "Steepest Descent" via parallel evaluation.
    # For each position i, if we flip it, how much does the max abs sum change?
    # This is hard to compute exactly for all i at once because the max changes non-linearly.
    # However, we can use a "Greedy Batch Flip" approach:
    # 1. Identify "bad" progressions (those with high abs sum).
    # 2. Try to flip signs that contribute to these bad progressions to reduce their sum.
    
    # Alternative Structural Change: **Iterated Local Search with Batch Flips**
    # We will use a "Vectorized Greedy" approach where we evaluate the impact of flipping
    # ALL positions that are "aligned" with the current worst progression.
    
    # Let's try a different structural approach: **Genetic Algorithm / Crossover**
    # or **Tabu Search**.
    # Actually, the most effective structural change for this specific problem (min max constraint)
    # is often **Local Search with Multi-Point Moves** or **Simulated Annealing with large moves**.
    
    # Let's implement a **Vectorized Greedy Descent** with **Batch Flips**.
    # Idea: 
    # 1. Compute sums.
    # 2. Find the progression p* with max abs sum.
    # 3. For each position i in p*, check if flipping sign[i] reduces the sum of p* (and hopefully doesn't break others too much).
    # 4. To be more robust, we can evaluate the "delta" of flipping each i in the current worst progression.
    
    # But the current program does single flips. A material difference is **Batch Flipping**.
    # We will flip a random subset of size K (e.g., K=10, 20, 50) and accept if it improves.
    # This allows escaping local minima that single flips cannot.
    
    K_values = [5, 10, 20, 50, 100]
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        signs = rng.choice([-1, 1], size=n)
        sums = compute_sums(signs)
        current_score = max_abs_sum(sums)
        
        if current_score < best_score:
            best_signs = signs.copy()
            best_score = current_score
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try different batch sizes
            # We pick random subsets of indices to flip
            # To make it faster, we can try flipping all indices that are "bad"
            
            # Strategy: 
            # 1. Find the top few worst progressions.
            # 2. For each position i, calculate the "potential" improvement if flipped.
            #    This is expensive if done naively.
            #    
            # Let's stick to a simpler but effective batch flip:
            # Randomly select K indices, flip them, check if score improves.
            # If yes, accept. If no, try smaller K or different set.
            
            # To be efficient, we will do a "Greedy Batch" step:
            # 1. Identify the worst progression p_worst.
            # 2. For each index i in p_worst, calculate the new sum of p_worst if sign[i] is flipped.
            # 3. Flip the index i that gives the largest reduction in |sum[p_worst]|.
            # 4. Check if the global max abs sum decreased.
            # 5. Repeat.
            
            # This is a "Steepest Descent" on the worst progression.
            
            if current_score <= 1:
                break # Can't do better than 1 (or 0 if possible, but signs are +-1 so sum of 1 element is 1)
                # Actually, singleton APs have sum +-1. So min max abs sum is at least 1.
                # If we reach 1, we are optimal.
            
            # Find worst progression
            abs_sums = np.abs(sums)
            p_worst = np.argmax(abs_sums)
            
            # Get indices of this progression
            # We need a reverse mapping: prog_id -> list of indices
            # We built progs, so progs[p_worst] gives the indices.
            idxs = progs[p_worst]
            
            # Try flipping each index in this progression to see which one reduces the sum most
            # sum_p = sum(signs[i] for i in idxs)
            # If we flip i, new_sum_p = sum_p - 2 * signs[i]
            # We want to minimize |new_sum_p|
            
            current_sum_p = sums[p_worst]
            
            best_i = -1
            best_new_sum_p = current_sum_p
            best_delta_global = float('inf')
            
            # Evaluate flipping each i in idxs
            # We can vectorize this part?
            # signs_in_prog = signs[idxs]
            # new_sums_p = current_sum_p - 2 * signs_in_prog
            # We want the new_sum_p closest to 0.
            
            signs_in_prog = signs[idxs]
            new_sums_p = current_sum_p - 2 * signs_in_prog
            abs_new_sums_p = np.abs(new_sums_p)
            
            # Find the index in idxs that gives the smallest abs_new_sum_p
            # If there are ties, we might want to check global impact.
            # For now, pick the one that minimizes abs_new_sums_p.
            
            min_abs_idx_in_prog = np.argmin(abs_new_sums_p)
            i_to_flip = idxs[min_abs_idx_in_prog]
            
            # Calculate global impact of flipping i_to_flip
            old_sign = signs[i_to_flip]
            new_sign = -old_sign
            
            # Update sums
            delta_vec = 2 * new_sign * M[i_to_flip]
            sums_new = sums + delta_vec
            
            new_score = max_abs_sum(sums_new)
            
            if new_score < current_score:
                # Accept
                signs[i_to_flip] = new_sign
                sums = sums_new
                current_score = new_score
                improved = True
                
                if current_score < best_score:
                    best_signs = signs.copy()
                    best_score = current_score
            else:
                # No improvement with the "best" local move. 
                # Try a random flip to escape? Or try the second best local move?
                # To keep it structural and fast, let's try a random flip if stuck.
                # Or, break the local search and restart with a perturbation.
                # Let's do a small perturbation: flip 5 random signs.
                if time.time() < deadline - 2:
                    flip_idx = rng.choice(n, size=5, replace=False)
                    signs[flip_idx] *= -1
                    sums = compute_sums(signs)
                    current_score = max_abs_sum(sums)
                    if current_score < best_score:
                        best_signs = signs.copy()
                        best_score = current_score
                    # We set improved=True to continue the loop, but this is a random walk step.
                    # To prevent infinite loops of random walks, we should limit this.
                    # Let's just break the local search if no greedy improvement is found.
                    break

    # Final Polish: Single flip greedy
    if time.time() < deadline - 1 and best_signs is not None:
        signs = best_signs.copy()
        sums = compute_sums(signs)
        current_score = max_abs_sum(sums)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1:
                    break
                old_sign = signs[i]
                new_sign = -old_sign
                delta_vec = 2 * new_sign * M[i]
                sums_new = sums + delta_vec
                new_score = max_abs_sum(sums_new)
                
                if new_score < current_score:
                    signs[i] = new_sign
                    sums = sums_new
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_signs = signs.copy()
                        best_score = current_score
                    # Restart loop? Or just continue? 
                    # Restarting is safer for steepest descent.
                    break # Restart the for loop to re-evaluate from start? 
                    # Actually, just continue to next i is faster, but might miss better global moves.
                    # Given time constraints, just continue.
        
        if current_score < best_score:
            best_signs = signs.copy()
            best_score = current_score

    return [int(x) for x in best_signs]
