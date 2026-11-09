import time
from random import Random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
        
    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def fast_solve(vals):
        n = len(vals)
        rng = Random(42)
        
        # Helper: Check if val can be added to set of values
        def can_add(val, s):
            # s is a set of values
            # Condition: no a, b in s such that a+b=val, a+val=b, val+b=a
            # 1. a + val = b => b - val = a => b in s and b-val in s
            # 2. val + b = a => a - val = b => a in s and a-val in s (same as 1)
            # 3. a + b = val => a in s and val-a in s
            
            # Check 1 & 2: exists x in s such that x + val in s
            # This covers a+val=b and val+b=a (symmetric)
            for x in s:
                if x + val in s:
                    return False
            
            # Check 3: exists x in s such that x + (val-x) = val => val-x in s
            # So exists x in s such that val-x in s
            # Note: if x = val-x, then 2x=val, so x=val/2.
            # The loop above checks x+val in s.
            # Here we check if val can be formed by sum of two elements in s.
            for x in s:
                if val - x in s:
                    return False
            
            return True

        # Helper: Greedy fill
        def greedy_fill(order, current_set, current_vals):
            # current_set: set of indices
            # current_vals: set of values
            # order: list of indices to try adding
            for idx in order:
                if idx in current_set:
                    continue
                val = vals[idx]
                if can_add(val, current_vals):
                    current_set.add(idx)
                    current_vals.add(val)
            return current_set, current_vals

        best_set = set()
        best_vals = set()
        best_size = 0
        
        # 1. Multiple Greedy Initializations
        strategies = [
            sorted(range(n), key=lambda i: abs(vals[i]), reverse=True),
            sorted(range(n), key=lambda i: abs(vals[i])),
            sorted(range(n), key=lambda i: vals[i], reverse=True),
            sorted(range(n), key=lambda i: vals[i]),
            list(range(n)),
            list(range(n-1, -1, -1))
        ]
        
        # Add a few random permutations
        for _ in range(5):
            perm = list(range(n))
            rng.shuffle(perm)
            strategies.append(perm)
            
        for order in strategies:
            curr_set = set()
            curr_vals = set()
            curr_set, curr_vals = greedy_fill(order, curr_set, curr_vals)
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = curr_set
                best_vals = curr_vals
                
        # 2. Local Search
        # We will iterate until no improvement or time runs out.
        # Moves:
        # A. Remove one element, refill greedily (randomized order for diversity)
        # B. Swap: Remove one element u, try to add one element v from complement.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_set = set(best_set)
            current_vals = set(best_vals)
            
            # List of elements in the set to consider for removal
            elements_in_set = list(current_set)
            rng.shuffle(elements_in_set) # Randomize order to escape symmetry
            
            for u_idx in elements_in_set:
                if time.time() > deadline - 2.0:
                    break
                
                u_val = vals[u_idx]
                
                # --- Move A: Remove and Refill ---
                # Try a few random refill orders
                for refill_iter in range(3):
                    temp_set = current_set - {u_idx}
                    temp_vals = current_vals - {u_val}
                    
                    # Candidates are all indices not in temp_set
                    candidates = [i for i in range(n) if i not in temp_set]
                    # Randomize candidates for diversity
                    rng.shuffle(candidates)
                    
                    new_set, new_vals = greedy_fill(candidates, set(temp_set), set(temp_vals))
                    
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_vals = new_vals
                        best_size = len(new_set)
                        improved = True
                        break # Restart local search with new best
                    
                if improved:
                    break # Restart local search with new best
                
                # --- Move B: Swap ---
                # Remove u, try to add a specific v that might fit now.
                # Which v? Those that were blocked by u or are "close" in value?
                # For simplicity, try a random subset of candidates or all candidates if N is small.
                # For N=240, trying all candidates for each u is 240*240 = 57600 checks.
                # Each check is O(|S|) ~ 80. Total ~ 4.6M ops. This is fast in Python if optimized.
                
                temp_set = current_set - {u_idx}
                temp_vals = current_vals - {u_val}
                
                # We want to find a v not in temp_set such that can_add(v, temp_vals) is True
                # AND adding v allows us to keep the size (swap) or increase it.
                # Actually, if we just add v, size increases by 1 (since we removed u).
                # So if we find ANY valid v, we improve size by 0 (swap) or 1 (if we don't remove u? No, we did remove u).
                # Wait. If we remove u and add v, size is same.
                # We want size to increase.
                # So we should remove u and try to add MULTIPLE elements.
                # That's what Move A does (refill).
                # Move B is useful if Move A fails to find a better set because the greedy refill order is bad.
                # But Move A with randomization should cover this.
                
                # Let's try a "2-swap" or just rely on Move A with more iterations.
                # Let's increase the refill iterations in Move A if time permits.
                # Or, try to add a single element v to the original set without removing u?
                # That's impossible if the set is maximal.
                
                # Let's try a different move: 
                # Remove u, add v, then refill.
                # This is essentially Move A but starting from a different state?
                # No, Move A removes u and refills from scratch.
                
                # What if we remove TWO elements and refill?
                # That's expensive.
                
                # Let's stick to Move A with more random restarts of the refill order.
                # Increase refill_iter to 5 or 10.
        
        return [vals[i] for i in sorted(best_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        results.append(fast_solve(vals))

    return results
