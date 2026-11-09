import time
from random import Random
import sys
import math

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

    def solve_instance(vals, time_budget):
        n = len(vals)
        if n == 0:
            return []
        
        def can_add(val, current_vals):
            for x in current_vals:
                if val + x in current_vals:
                    return False
                if val - x in current_vals:
                    return False
            return True

        def greedy_fill(order, start_set=None, start_vals=None):
            if start_set is None:
                current_set = set()
                current_vals = set()
            else:
                current_set = set(start_set)
                current_vals = set(start_vals)
            
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
        
        rng_init = Random(12345)
        
        order1 = sorted(range(n), key=lambda i: abs(vals[i]), reverse=True)
        s, v = greedy_fill(order1)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        pos_idx = [i for i in range(n) if vals[i] > 0]
        neg_idx = [i for i in range(n) if vals[i] < 0]
        order2 = sorted(pos_idx, key=lambda i: vals[i], reverse=True) + \
                 sorted(neg_idx, key=lambda i: vals[i])
        s, v = greedy_fill(order2)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        for _ in range(5):
            order3 = list(range(n))
            rng_init.shuffle(order3)
            s, v = greedy_fill(order3)
            if len(s) > best_size:
                best_size = len(s)
                best_set, best_vals = s, v

        current_set = set(best_set)
        current_vals = set(best_vals)
        current_size = best_size
        
        T = 1.0
        alpha = 0.995
        min_T = 1e-6
        rng_sa = Random(54321)
        
        sa_start = time.time()
        sa_deadline = sa_start + time_budget * 0.85
        
        while time.time() < sa_deadline:
            if T < min_T:
                break
            
            if not current_set:
                break
                
            set_list = list(current_set)
            remove_idx = rng_sa.choice(set_list)
            remove_val = vals[remove_idx]
            
            temp_set = current_set - {remove_idx}
            temp_vals = current_vals - {remove_val}
            
            fill_order = list(range(n))
            rng_sa.shuffle(fill_order)
            
            new_set, new_vals = greedy_fill(fill_order, start_set=temp_set, start_vals=temp_vals)
            
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
                prob = math.exp(delta / T)
                if rng_sa.random() < prob:
                    current_set = new_set
                    current_vals = new_vals
                    current_size = new_size
            
            T *= alpha
        
        final_set = set(best_set)
        final_vals = set(best_vals)
        
        missing = [i for i in range(n) if i not in final_set]
        missing.sort(key=lambda i: abs(vals[i]), reverse=True)
        for idx in missing:
            if can_add(vals[idx], final_vals):
                final_set.add(idx)
                final_vals.add(vals[idx])
        
        missing = [i for i in range(n) if i not in final_set]
        pos_missing = sorted([i for i in missing if vals[i] > 0], key=lambda i: vals[i], reverse=True)
        neg_missing = sorted([i for i in missing if vals[i] < 0], key=lambda i: vals[i])
        for idx in pos_missing + neg_missing:
            if can_add(vals[idx], final_vals):
                final_set.add(idx)
                final_vals.add(vals[idx])
        
        rng_polish = Random(99999)
        for _ in range(10):
            missing = [i for i in range(n) if i not in final_set]
            if not missing:
                break
            rng_polish.shuffle(missing)
            for idx in missing:
                if can_add(vals[idx], final_vals):
                    final_set.add(idx)
                    final_vals.add(vals[idx])
        
        return [vals[i] for i in sorted(final_set)]

    results = []
    time_alloc = {60: 20.0, 120: 30.0, 240: 50.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        budget = min(time_alloc.get(size, 30.0), remaining - 5.0)
        if budget < 5.0:
            budget = max(5.0, remaining - 5.0)
        
        vals = instance(size, seed)
        results.append(solve_instance(vals, budget))

    return results
