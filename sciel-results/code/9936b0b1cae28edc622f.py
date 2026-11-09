import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def sum_free_check(S):
        """Return True if S is sum-free (no a+b=c with a,b,c in S, a=b allowed)."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def build_sum_free(arr):
        n = len(arr)
        if n == 0:
            return []
        
        elem_to_idx = {v: i for i, v in enumerate(arr)}
        rng = random.Random(42)
        
        # Precompute for each index i, the set of indices j such that arr[i] + arr[j] is in arr
        # This helps in fast checking? 
        # Actually, for can_add(idx, current_set):
        # We need to check if arr[idx] + arr[k] is in current_set for any k in current_set.
        # And if arr[idx] - arr[k] is in current_set for any k in current_set.
        
        # Let's stick to the O(|S|) check which is fast for N=240.
        
        def can_add(idx, current):
            """Check if idx can be added to current without violating sum-free."""
            c_val = arr[idx]
            # Check 1: for all a in current, c_val - a is not in current
            # This is equivalent to: no a, b in current such that a + b = c_val
            for a_idx in current:
                a_val = arr[a_idx]
                b_val = c_val - a_val
                if b_val in elem_to_idx and elem_to_idx[b_val] in current:
                    return False
            # Check 2: for all b in current, c_val + b is not in current
            # This is equivalent to: no a, b in current such that a + b = c_val? No.
            # If we add c_val, we must ensure c_val + b != a for any a,b in current.
            # And a + c_val != b for any a,b in current.
            # Since addition is commutative, Check 1 covers a+b=c_val.
            # Check 2 covers c_val + b = a => a - b = c_val.
            # So we need:
            # 1. No a,b in current such that a+b = c_val
            # 2. No a,b in current such that a-b = c_val (which implies a = b + c_val)
            
            # My previous code had:
            # Check 1: c_val - a in current (implies a + (c_val-a) = c_val). Correct.
            # Check 2: c_val + b in current (implies b + c_val = (c_val+b)). Correct.
            
            for b_idx in current:
                b_val = arr[b_idx]
                s_val = c_val + b_val
                if s_val in elem_to_idx and elem_to_idx[s_val] in current:
                    return False
            return True

        def greedy_init(order):
            included = set()
            for idx in order:
                if can_add(idx, included):
                    included.add(idx)
            return included

        # Multiple greedy starts
        orders = [
            sorted(range(n), key=lambda i: -abs(arr[i])),
            sorted(range(n), key=lambda i: abs(arr[i])),
            list(range(n))
        ]
        rng.shuffle(orders[2])
        
        best_included = set()
        for order in orders:
            inc = greedy_init(order)
            if len(inc) > len(best_included):
                best_included = inc

        included = set(best_included)
        
        improved = True
        # Local Search Loop
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Phase 1: Try adding any excluded element
            excluded = [i for i in range(n) if i not in included]
            rng.shuffle(excluded)
            for idx in excluded:
                if time.time() >= deadline - 10.0:
                    break
                if can_add(idx, included):
                    included.add(idx)
                    improved = True
                    # Restart excluded list? Or just continue?
                    # If we add one, the set of valid additions changes.
                    # It's safer to restart the loop or re-scan.
                    # For efficiency, we can just break and restart the while loop.
                    break
            
            if improved:
                continue # Restart to check for more additions
            
            # Phase 2: Kick and Re-fill
            # Remove a random element, then greedily fill
            if len(included) > 0:
                kick_candidates = list(included)
                rng.shuffle(kick_candidates)
                # Try a few kicks
                for kick_idx in kick_candidates[:10]: # Limit kicks per iteration
                    if time.time() >= deadline - 10.0:
                        break
                    
                    trial = set(included)
                    trial.remove(kick_idx)
                    
                    # Greedily add from excluded
                    excluded_now = [i for i in range(n) if i not in trial]
                    rng.shuffle(excluded_now)
                    
                    added_any = False
                    for add_idx in excluded_now:
                        if time.time() >= deadline - 10.0:
                            break
                        if can_add(add_idx, trial):
                            trial.add(add_idx)
                            added_any = True
                    
                    if len(trial) > len(included):
                        included = trial
                        improved = True
                        break # Restart local search with new best
        
        # Final check
        result = [arr[i] for i in included]
        if not sum_free_check(result):
            # Fallback
            result = [x for x in arr if x % 2]
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() >= deadline - 5.0:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        sf = build_sum_free(vals)
        if not sum_free_check(sf):
            sf = [x for x in vals if x % 2]
        results.append(sf)

    return results
