import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        # Optimization: check a+b=c. 
        # Since a,b,c are in subset, we can just iterate pairs.
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                b = subset[j]
                c = a + b
                if c in s_set:
                    return False
        return True

    def can_add(C_set, x):
        """Check if x can be added to C (represented as set) to maintain sum-free property."""
        # We need to ensure:
        # 1. x + y != z for any y, z in C
        # 2. y + x != z for any y, z in C (same as above)
        # 3. x + x != z for any z in C
        # 4. x + y != x => y != 0 (0 not in C)
        # 5. y + x != x => y != 0
        # 6. x + x != x => x != 0 (x is nonzero)
        
        # So we check:
        # For all y in C:
        #   (x + y) not in C_set
        #   (x - y) not in C_set  (this covers y + x = z where z=x-y? No. 
        #                         # a+b=c. If a=x, b=y, c=x+y. 
        #                         # If a=y, b=x, c=y+x. Same.
        #                         # If a=x, b=x, c=2x.
        #                         # If a=y, b=z, c=x => y+z=x => z=x-y. 
        #                         # So we must check if x-y is in C for any y in C.
        
        for y in C_set:
            if (x + y) in C_set:
                return False
            if (x - y) in C_set:
                return False
        if (2 * x) in C_set:
            return False
        return True

    def greedy_add(S, C_init):
        """Greedily add elements from S to C_init until no more can be added."""
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        
        # We iterate until no change
        changed = True
        while changed:
            changed = False
            new_excluded = []
            for x in excluded:
                if can_add(C_set, x):
                    C.append(x)
                    C_set.add(x)
                    changed = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        return C

    def local_search(S, C_init, max_iters=100):
        """
        Local search to maximize sum-free subset size.
        Combines greedy add and 1-2 swaps.
        """
        C = list(C_init)
        C_set = set(C)
        
        # First, saturate with greedy add
        C = greedy_add(S, C)
        C_set = set(C)
        
        best_C = C
        best_len = len(C)
        
        rng = random.Random(42)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            improved = False
            excluded = [x for x in S if x not in C_set]
            
            # Try 1-2 swaps: remove u from C, add v, w from excluded
            # To be efficient, we only try a subset of swaps per iteration
            # or just one random swap attempt
            
            # 1. Try to find a beneficial 1-2 swap
            # We pick a random u from C to remove
            if C:
                u_idx = rng.randrange(len(C))
                u = C[u_idx]
                C_temp = C[:u_idx] + C[u_idx+1:]
                C_temp_set = set(C_temp)
                
                # Try to add two elements from excluded to C_temp
                # We sort excluded by some heuristic (e.g., abs value) to find good candidates faster?
                # Or just random sample.
                # Let's try to add as many as possible from excluded to C_temp greedily, 
                # but we specifically want to see if we can get 2 added.
                
                added = []
                # To find 2 adds, we can just run a mini greedy add on C_temp with excluded
                # But we only care if size increases by 1 (remove 1, add 2 => +1 net)
                
                # Optimization: Check if any pair (v, w) from excluded can be added to C_temp
                # This is O(|excluded|^2 * |C|). Too slow for large N.
                # Alternative: Greedily add to C_temp. If we can add >= 2, it's a good move.
                
                C_temp_new = list(C_temp)
                C_temp_new_set = set(C_temp_new)
                for x in excluded:
                    if can_add(C_temp_new_set, x):
                        C_temp_new.append(x)
                        C_temp_new_set.add(x)
                        
                if len(C_temp_new) > len(C):
                    C = C_temp_new
                    C_set = C_temp_new_set
                    improved = True
                    if len(C) > best_len:
                        best_C = C
                        best_len = len(C)
                    break # Restart the loop to re-saturate and try new swaps
                
            # If no improvement via swap, try a perturbation (random remove)
            if not improved:
                # Remove 1 or 2 random elements to escape local optimum
                if len(C) > 2:
                    remove_count = rng.choice([1, 2])
                    indices_to_remove = rng.sample(range(len(C)), remove_count)
                    indices_to_remove.sort(reverse=True)
                    for idx in indices_to_remove:
                        C.pop(idx)
                    C_set = set(C)
                else:
                    # If C is very small, reset to a random structural candidate?
                    # Or just break
                    break
            
        return best_C

    def solve_instance(S, time_budget):
        n = len(S)
        
        # Generate initial candidates
        candidates = []
        
        # 1. Odds
        S_odd = [x for x in S if x % 2 != 0]
        candidates.append(S_odd)
        
        # 2. Large Positive
        M = max(S)
        S_large_pos = [x for x in S if x > M / 2.0]
        candidates.append(S_large_pos)
        
        # 3. Large Negative
        m = min(S)
        S_large_neg = [x for x in S if x < m / 2.0]
        candidates.append(S_large_neg)
        
        # 4. Mod 3 = 1
        S_mod1 = [x for x in S if x % 3 == 1]
        candidates.append(S_mod1)
        
        # 5. Mod 3 = 2
        S_mod2 = [x for x in S if x % 3 == 2]
        candidates.append(S_mod2)
        
        # 6. Greedy by abs desc
        S_sorted_abs = sorted(S, key=lambda x: abs(x), reverse=True)
        C_greedy_abs = []
        C_set = set()
        for x in S_sorted_abs:
            if can_add(C_set, x):
                C_greedy_abs.append(x)
                C_set.add(x)
        candidates.append(C_greedy_abs)
        
        # 7. Greedy by abs asc
        S_sorted_abs_asc = sorted(S, key=lambda x: abs(x))
        C_greedy_abs_asc = []
        C_set = set()
        for x in S_sorted_abs_asc:
            if can_add(C_set, x):
                C_greedy_abs_asc.append(x)
                C_set.add(x)
        candidates.append(C_greedy_abs_asc)

        # 8. Random Greedy
        rng_init = random.Random(123)
        S_shuffled = list(S)
        rng_init.shuffle(S_shuffled)
        C_greedy_rand = []
        C_set = set()
        for x in S_shuffled:
            if can_add(C_set, x):
                C_greedy_rand.append(x)
                C_set.add(x)
        candidates.append(C_greedy_rand)

        # Evaluate candidates and pick the best starting point for local search
        best_start = []
        best_start_len = 0
        for C in candidates:
            if len(C) > best_start_len:
                # Quick check if it's valid (should be by construction, but safe)
                if is_sum_free(C):
                    best_start = C
                    best_start_len = len(C)
        
        # Run local search from the best start
        # Also run from a few other good starts to diversify
        final_best = best_start
        final_best_len = best_start_len
        
        # Time management
        # Split time among a few restarts
        num_restarts = 3
        time_per_restart = time_budget / num_restarts
        
        for i in range(num_restarts):
            if time.time() > start_time + time_budget:
                break
            
            # For the first restart, use the best structural candidate
            # For subsequent restarts, use the result of the previous local search 
            # or a different structural candidate to explore different basins
            
            if i == 0:
                start_C = best_start
            else:
                # Pick a different structural candidate or perturb the previous best
                # Let's pick the next best structural candidate that isn't best_start
                start_C = candidates[i % len(candidates)]
                # If that's the same as best_start, just use best_start
                if start_C == best_start and i > 0:
                    start_C = best_start
            
            # Run local search with a time limit or iteration limit
            # Since N is small, we can run many iterations
            improved_C = local_search(S, start_C, max_iters=200)
            
            if len(improved_C) > final_best_len:
                final_best = improved_C
                final_best_len = len(improved_C)
                
            # If we improved, use the improved solution as a base for the next restart?
            # Or keep restarting from structural candidates to avoid getting stuck?
            # Restarting from structural candidates is safer for diversity.
            
        return final_best

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time: give more to larger instances
    # Total 145s.
    # 60: 20s, 120: 40s, 240: 80s. Total 140s.
    time_alloc = [20, 40, 80]
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        C = solve_instance(S, time_alloc[i])
        
        # Final safety check
        if not is_sum_free(C):
            C = [x for x in S if x % 2 != 0]
        
        results.append(C)
    
    return results
