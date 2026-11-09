import time
from random import Random
import sys
import numpy as np

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
        
    def fast_solve(vals):
        n = len(vals)
        
        # Precompute: for each value v, the set of values x such that v+x is in vals
        # and v-x is in vals. This allows fast conflict checking.
        val_set = set(vals)
        
        # For each index i, precompute:
        # partners_plus[i] = set of indices j such that vals[i] + vals[j] in val_set
        # partners_minus[i] = set of indices j such that vals[i] - vals[j] in val_set (i.e., vals[j] - vals[i] in val_set)
        # Actually, for can_add(v, S): we need to check if there exists x in S such that v+x in S OR v-x in S
        # So for a given v, the "forbidden" elements are those x where v+x in S or v-x in S
        # We can precompute for each v: conflict_with[v] = set of values x such that v+x in val_set or v-x in val_set
        # But this depends on S, so we need a different approach.
        
        # Alternative: maintain a set of current values S. To check if v can be added:
        # Check if any x in S has v+x in S or v-x in S.
        # We can precompute for each v: the list of values x such that v+x is in the full universe.
        # Then check if any of those x are in S AND v+x is in S.
        
        # Let's precompute: for each value v, sum_partners[v] = {x : v+x in val_set}
        # and diff_partners[v] = {x : v-x in val_set} i.e. {x : x = v - y for y in val_set} = {v-y : y in val_set}
        
        sum_partners = {}
        diff_partners = {}
        for v in vals:
            sp = set()
            dp = set()
            for u in vals:
                if v + u in val_set:
                    sp.add(u)
                if v - u in val_set:
                    dp.add(u)
            sum_partners[v] = sp
            diff_partners[v] = dp
        
        def can_add(val, current_set):
            """Check if val can be added to current_set (a set of values) without violating sum-free."""
            # Check if there exists x in current_set such that val+x in current_set
            for x in current_set:
                if val + x in current_set:
                    return False
                if val - x in current_set:
                    return False
            return True
        
        def greedy_fill(order, current_set=None):
            """Greedy fill given an order of indices."""
            if current_set is None:
                s = set()
            else:
                s = set(current_set)
            
            for idx in order:
                val = vals[idx]
                if val not in s:
                    if can_add(val, s):
                        s.add(val)
            return s
        
        # Multiple greedy strategies
        strategies = []
        
        # Strategy 1: All positives (they form a sum-free set if we're careful)
        # Actually, positives alone: if a,b,c > 0, a+b=c is possible. So not necessarily sum-free.
        # But large positives are less likely to have a+b=c in the set.
        
        # Strategy 1: Descending absolute value
        order1 = sorted(range(n), key=lambda i: abs(vals[i]), reverse=True)
        strategies.append(order1)
        
        # Strategy 2: Positive values first (descending), then negative (ascending)
        pos_idx = [i for i in range(n) if vals[i] > 0]
        neg_idx = [i for i in range(n) if vals[i] < 0]
        order2 = sorted(pos_idx, key=lambda i: vals[i], reverse=True) + \
                 sorted(neg_idx, key=lambda i: vals[i])
        strategies.append(order2)
        
        # Strategy 3: Negative values first (ascending), then positive (descending)
        order3 = sorted(neg_idx, key=lambda i: vals[i]) + \
                 sorted(pos_idx, key=lambda i: vals[i], reverse=True)
        strategies.append(order3)
        
        # Strategy 4: Random shuffles
        rng = Random(12345)
        for _ in range(20):
            order_r = list(range(n))
            rng.shuffle(order_r)
            strategies.append(order_r)
        
        # Find best greedy solution
        best_set = set()
        best_size = 0
        
        for order in strategies:
            if time.time() > deadline - 20:
                break
            current = greedy_fill(order)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current
        
        # Local Search: try 1-swap moves (remove one, add one) and 2-swap (remove one, add two)
        # Also try remove-and-refill for better escapes
        
        improved = True
        iteration = 0
        max_iterations = 50
        
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            # Try each removal
            removal_order = list(best_set)
            rng.shuffle(removal_order)
            
            for v_idx in removal_order:
                if time.time() > deadline - 10:
                    break
                
                v_val = vals[v_idx]
                temp_set = best_set - {v_val}
                
                # Try to add one element from the complement
                # Find candidates that can be added
                candidates = []
                for idx in range(n):
                    if idx in best_set:
                        continue
                    val = vals[idx]
                    if can_add(val, temp_set):
                        candidates.append(idx)
                
                # Try each candidate addition
                for c_idx in candidates:
                    new_set = temp_set | {vals[c_idx]}
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(new_set)
                        improved = True
                        break
                
                if improved:
                    break
            
            # Also try remove-and-refill for better escapes
            if not improved and time.time() < deadline - 15:
                for v_idx in removal_order[:5]:  # Try a few removals
                    if time.time() > deadline - 10:
                        break
                    v_val = vals[v_idx]
                    temp_set = best_set - {v_val}
                    
                    # Refill greedily
                    order_fill = list(range(n))
                    rng.shuffle(order_fill)
                    new_set = greedy_fill(order_fill, temp_set)
                    
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(new_set)
                        improved = True
                        break
        
        return [vals[i] for i in sorted(best_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if time runs out
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        results.append(fast_solve(vals))

    return results
