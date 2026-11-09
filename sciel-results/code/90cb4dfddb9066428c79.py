"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    _t_start = time.time()
    _deadline = 120.0  # hard kill at 150s, keep safety margin

    def _time_left():
        return _deadline - (time.time() - _t_start)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if subset is sum-free (no a+b=c with a,b,c in subset, a=b allowed)."""
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def fast_solve(values, seed_offset=0):
        """
        Greedy + local search for maximum sum-free subset.
        Strategy:
        1. Build conflict graph implicitly.
        2. Greedy: start with all elements, iteratively remove the element
           involved in the most conflicts (or use a degree-based heuristic).
        3. Local search: try adding back removed elements if they don't conflict,
           or swap operations.
        """
        n = len(values)
        idx_map = {v: i for i, v in enumerate(values)}
        
        # Precompute for each pair (i,j) whether values[i]+values[j] is in the set
        # conflict[i][j] = index k such that values[i]+values[j]==values[k], or -1
        # But we need: for a given subset, check if it's sum-free.
        
        # Alternative approach: 
        # An element c conflicts with the set if there exist a,b in set with a+b=c.
        # For greedy removal: start with all, find violations, remove one of a,b,c.
        
        # Better: use the fact that if we sort, we can think about it differently.
        # Let's use a simulated annealing / local search approach.
        
        # Initial solution: greedy by magnitude (take large elements first)
        # Large positive + large positive = even larger (likely not in set)
        # Large negative + large negative = even more negative (likely not in set)
        
        # Strategy: partition into "safe" groups.
        # Key insight: if we take all elements with |x| > M, then x+y has |x+y| > M/2 
        # but might still be in range. However, if we take elements with x > B or x < -B,
        # then x+y > 2B or x+y < -2B, which are outside the original range if B is large enough.
        
        # Actually, let's use a different approach:
        # 1. Start with a good initial solution (e.g., all positives or all negatives, or large magnitude)
        # 2. Try to add elements from the remaining pool if they don't create conflicts
        # 3. Local search: swap operations
        
        # Initial solution: all elements with x > 0 (positives are sum-free if no a+b=c among them)
        # Wait, positives are NOT necessarily sum-free: 1+2=3.
        
        # Better initial: take elements with largest absolute value.
        # If x,y have large |.|, then x+y has even larger |.|, likely not in the set.
        
        # Let's try: sort by |x| descending, greedily add if sum-free.
        
        def check_conflict_add(subset_set, val):
            """Check if adding val to subset_set creates a conflict."""
            # Need: no a,b in subset_set such that a+b=val
            # And: no a in subset_set such that a+val is in subset_set (i.e., val+a=c)
            # And: 2*val not in subset_set (a=b=val case)
            # Actually all three are covered by: for all a in subset_set, a+val not in subset_set ∪ {val}
            # Wait: conflict if ∃ a,b in (subset_set ∪ {val}) with a+b=c in (subset_set ∪ {val})
            # New conflicts only involve val:
            # 1. val + a = c for some a,c in subset_set
            # 2. a + b = val for some a,b in subset_set  
            # 3. val + val = c for some c in subset_set
            # 4. val + a = val => a=0, but 0 not in set
            # So check:
            # - For each a in subset_set: if val+a in subset_set, conflict
            # - For each a in subset_set: if a+val in subset_set, same as above
            # - For each a,b in subset_set: if a+b==val, conflict
            # - If 2*val in subset_set, conflict
            
            # More efficient:
            # For a in subset_set: if (val + a) in subset_set: return True
            # For a in subset_set: if (val - a) in subset_set: return True  (a + (val-a) = val)
            # if 2*val in subset_set: return True
            
            for a in subset_set:
                if (val + a) in subset_set:
                    return True
                if (val - a) in subset_set:
                    return True
            if 2 * val in subset_set:
                return True
            return False
        
        # Greedy construction: sort by |x| descending
        sorted_by_mag = sorted(values, key=lambda x: abs(x), reverse=True)
        
        subset = set()
        for v in sorted_by_mag:
            if not check_conflict_add(subset, v):
                subset.add(v)
        
        # Local search: try to add back removed elements
        removed = [v for v in values if v not in subset]
        
        # Try adding each removed element
        improved = True
        while improved and _time_left() > 5:
            improved = False
            for v in removed:
                if v not in subset and not check_conflict_add(subset, v):
                    subset.add(v)
                    removed.remove(v)
                    improved = True
        
        # Simulated annealing / random restarts for local search
        # Try swapping: remove one, add another
        rng = random.Random(seed_offset)
        best_subset = set(subset)
        best_size = len(best_subset)
        
        # Try multiple random restarts with different initial orderings
        for restart in range(20):
            if _time_left() < 5:
                break
            
            # Random shuffle of values
            shuffled = values[:]
            rng.shuffle(shuffled)
            
            subset2 = set()
            for v in shuffled:
                if not check_conflict_add(subset2, v):
                    subset2.add(v)
            
            # Try to add back
            removed2 = [v for v in values if v not in subset2]
            for v in removed2:
                if not check_conflict_add(subset2, v):
                    subset2.add(v)
            
            if len(subset2) > best_size:
                best_subset = set(subset2)
                best_size = len(subset2)
        
        return sorted(best_subset)

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if _time_left() < 10:
            # Fallback: use parity
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        result = fast_solve(vals, seed_offset=seed)
        
        # Verify sum-free
        if not is_sum_free(result):
            # Fallback to parity
            result = [x for x in vals if x % 2]
        
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
