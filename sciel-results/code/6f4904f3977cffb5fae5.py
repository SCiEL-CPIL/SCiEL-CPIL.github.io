import time
from random import Random
import sys
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_instance(vals):
        n = len(vals)
        if n == 0:
            return []
        
        # Build conflict graph
        # Two elements conflict if a + b = c for some c in the set
        # Actually, we need: no a, b, c in subset such that a + b = c
        # So for each pair (a, b), if a+b is in the set, then a and b cannot both be in the subset
        # AND a+b cannot be in the subset if both a and b are.
        # This is a 3-uniform hypergraph problem, but we can model it as:
        # For each c in vals, and each a in vals such that a != c and c-a in vals, 
        # we have a constraint: not (a in subset AND c-a in subset AND c in subset)
        
        # For branch and bound, let's use a different approach.
        # We'll use a greedy + local search approach with better heuristics.
        
        val_set = set(vals)
        
        def can_add(val, current_vals_set):
            """Check if val can be added to current_vals_set."""
            for x in current_vals_set:
                if val + x in current_vals_set:
                    return False
                if val - x in current_vals_set:
                    return False
            return True

        def greedy_fill(order, start_set=None, start_vals=None):
            """Greedy fill given an order of values."""
            if start_set is None:
                current_set = set()
                current_vals = set()
            else:
                current_set = set(start_set)
                current_vals = set(start_vals)
            
            for val in order:
                if val in current_vals:
                    continue
                if can_add(val, current_vals):
                    current_set.add(val)
                    current_vals.add(val)
            return current_set, current_vals

        # 1. Multiple Greedy Initializations
        best_set = set()
        best_vals = set()
        best_size = 0
        
        rng_init = Random(12345)
        
        # Strategy 1: Descending absolute value
        order1 = sorted(vals, key=lambda v: abs(v), reverse=True)
        s, v = greedy_fill(order1)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 2: Odds first (descending abs), then Evens (descending abs)
        odds = [v for v in vals if v % 2 != 0]
        evens = [v for v in vals if v % 2 == 0]
        order2 = sorted(odds, key=lambda v: abs(v), reverse=True) + \
                 sorted(evens, key=lambda v: abs(v), reverse=True)
        s, v = greedy_fill(order2)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 3: Random shuffles
        for _ in range(15):
            order3 = list(vals)
            rng_init.shuffle(order3)
            s, v = greedy_fill(order3)
            if len(s) > best_size:
                best_size = len(s)
                best_set, best_vals = s, v

        # Strategy 4: Positive first, then negative
        pos = sorted([v for v in vals if v > 0], key=lambda v: abs(v), reverse=True)
        neg = sorted([v for v in vals if v < 0], key=lambda v: abs(v), reverse=True)
        order4 = pos + neg
        s, v = greedy_fill(order4)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # 2. Simulated Annealing / Local Search
        current_set = set(best_set)
        current_vals = set(best_vals)
        current_size = best_size
        
        T = 1.0
        alpha = 0.995
        min_T = 1e-6
        
        rng_sa = Random(54321)
        
        refill_order = sorted(vals, key=lambda v: abs(v), reverse=True)
        
        while time.time() < deadline - 5.0:
            if T < min_T:
                break
            
            k = 1 if rng_sa.random() < 0.7 else 2
            
            if len(current_set) < k:
                break
            
            remove_vals = rng_sa.sample(list(current_set), k)
            
            temp_set = current_set - set(remove_vals)
            temp_vals = current_vals - set(remove_vals)
            
            new_set, new_vals = greedy_fill(refill_order, start_set=temp_set, start_vals=temp_vals)
            
            new_size = len(new_set)
            delta = new_size - current_size
            
            if delta > 0:
                current_set = new_set
                current_vals = new_vals
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_vals = set(current_vals)
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(delta / T)
                if rng_sa.random() < prob:
                    current_set = new_set
                    current_vals = new_vals
                    current_size = new_size
            
            T *= alpha

        # Final polish
        final_set, final_vals = greedy_fill(refill_order, start_set=best_set, start_vals=best_vals)
        
        return sorted(final_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        results.append(solve_instance(vals))

    return results
