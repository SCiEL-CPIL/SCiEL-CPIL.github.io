"""Rewritten program: Optimized Greedy + Simulated Annealing for Sum-Free Subset."""
# EVOLVE-BLOCK-START
import random
import math
import time

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(vals, seed_for_rng):
        n = len(vals)
        if n == 0:
            return []
        rng = random.Random(seed_for_rng * 1000003 + 7)
        val_set = set(vals)

        # Optimized can_add using set operations
        def can_add(S, x):
            # Check if adding x creates a sum a+b=x (where a,b in S)
            # This happens if there exists a in S such that x-a is also in S.
            # i.e., S intersects (x - S)
            # Note: x-S = {x-s for s in S}
            if S & {x - s for s in S}:
                return False
            # Check if adding x creates a sum a+x=b or x+b=a (where a,b in S)
            # This happens if there exists a in S such that a-x is in S (for a-x+b? No)
            # a+x=b => b-a=x. So we need b in S and b-x in S.
            # i.e., S intersects (S - x)
            if S & {s - x for s in S}:
                return False
            return True

        def is_sum_free(S):
            S_list = list(S)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    if (a + b) in S:
                        return False
            return True

        # 1. Greedy Initialization with multiple strategies
        best_greedy = set()
        
        # Strategy 1: Sort by degree (conflicts)
        # Degree of x: number of pairs (u,v) in vals such that u+v=x, u+x=v, v+x=u
        # We approximate this by counting how many other elements y satisfy:
        # y+x in vals OR y-x in vals OR x-y in vals
        degrees = {}
        for x in vals:
            deg = 0
            for y in vals:
                if y == x: continue
                if (x + y) in val_set: deg += 1
                if (x - y) in val_set: deg += 1
                if (y - x) in val_set: deg += 1
            degrees[x] = deg
        
        sorted_by_deg = sorted(vals, key=lambda x: degrees[x])
        
        # Strategy 2: Sort by absolute value (larger absolute values are often safer as sums go out of range or to less populated areas? 
        # Actually, for sum-free, taking the largest n/2 elements is a known lower bound strategy if they are positive. 
        # Here we have mixed signs. Taking elements with large |x| might be safer if the set is sparse.
        # Let's try sorting by abs value descending.
        sorted_by_abs = sorted(vals, key=lambda x: -abs(x))
        
        # Strategy 3: Random
        random_sorted = vals[:]
        rng.shuffle(random_sorted)

        strategies = [sorted_by_deg, sorted_by_abs, random_sorted]
        
        for strategy in strategies:
            S = set()
            for x in strategy:
                if can_add(S, x):
                    S.add(x)
            if len(S) > len(best_greedy):
                best_greedy = S

        cur = set(best_greedy)
        best = set(cur)
        best_size = len(cur)
        
        # 2. Simulated Annealing
        # We maintain a feasible state (sum-free).
        # Moves:
        # - Add: If valid, always accept.
        # - Remove: Accept with probability to escape local optima.
        # - Swap: Remove x, add y. If valid, accept.
        
        # Time limit check
        start_time = time.time()
        time_limit = 35.0 # Allow ~35s per instance, total ~105s < 120s
        
        T = 1.0
        cooling = 0.9995
        steps = 1000000 # High limit, break on time
        p_remove = 0.1
        
        # Precompute list of vals for random access
        vals_list = list(vals)
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            if T < 1e-6:
                break
                
            r = rng.random()
            
            if r < 0.4:
                # Try to Add
                # Pick random element not in cur
                # Optimization: Maintain a list of candidates? 
                # For small n (<=240), filtering is okay.
                candidates = [v for v in vals_list if v not in cur]
                if candidates:
                    x = rng.choice(candidates)
                    if can_add(cur, x):
                        cur.add(x)
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                        
            elif r < 0.8:
                # Try to Swap
                if cur:
                    x = rng.choice(list(cur))
                    candidates = [v for v in vals_list if v not in cur]
                    if candidates:
                        y = rng.choice(candidates)
                        # Check if (cur - {x}) + {y} is sum-free
                        # We can check can_add(cur - {x}, y)
                        # To save time, we can just try to add y to a temp set?
                        # No, can_add is fast.
                        temp_cur = cur - {x}
                        if can_add(temp_cur, y):
                            temp_cur.add(y)
                            # Accept swap
                            cur = temp_cur
                            # Note: Size is same, so best_size doesn't update here, 
                            # but cur is updated to a potentially better "shape".
                        
            else:
                # Try to Remove
                if cur:
                    x = rng.choice(list(cur))
                    # Accept removal with probability p_remove
                    if rng.random() < p_remove:
                        cur.remove(x)
                        
            T *= cooling
            
        # Final Verification
        if not is_sum_free(best):
             best = set(cur) # Fallback to current if best is invalid (shouldn't happen)
             
        return sorted(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        sub = solve_one(vals, seed)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
