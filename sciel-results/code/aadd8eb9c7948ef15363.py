"""Improved program for sum-free subset task."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 115  # seconds

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(full_set, deadline):
        n = len(full_set)
        if n == 0:
            return []
        
        full_sorted = sorted(full_set)
        best = None
        best_size = 0
        
        # Strategy 1: Take elements above various thresholds
        # If all elements > T and 2T > max, then sum-free
        max_val = full_sorted[-1]
        
        # Try different fraction thresholds
        for frac in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]:
            threshold = max_val * frac
            cand = [x for x in full_sorted if x > threshold]
            if cand and is_sum_free(cand) and len(cand) > best_size:
                best_size = len(cand)
                best = list(cand)
        
        # Strategy 2: Take positive elements above threshold
        pos = [x for x in full_sorted if x > 0]
        if pos:
            pos_max = max(pos)
            for frac in [0.5, 0.4, 0.3]:
                threshold = pos_max * frac
                cand = [x for x in pos if x > threshold]
                if cand and is_sum_free(cand) and len(cand) > best_size:
                    best_size = len(cand)
                    best = list(cand)
        
        # Strategy 3: Greedy add in order of decreasing value
        cand = []
        cand_set = set()
        for x in sorted(full_sorted, reverse=True):
            ok = True
            for a in cand:
                s = a + x
                if s in cand_set:
                    ok = False
                    break
            if ok and x + x not in cand_set:
                cand.append(x)
                cand_set.add(x)
        if cand and is_sum_free(cand) and len(cand) > best_size:
            best_size = len(cand)
            best = list(cand)
        
        # Strategy 4: Greedy add in order of increasing value
        cand = []
        cand_set = set()
        for x in full_sorted:
            ok = True
            for a in cand:
                s = a + x
                if s in cand_set:
                    ok = False
                    break
            if ok and x + x not in cand_set:
                cand.append(x)
                cand_set.add(x)
        if cand and is_sum_free(cand) and len(cand) > best_size:
            best_size = len(cand)
            best = list(cand)
        
        # Fallback: odd numbers
        cand_odd = [x for x in full_sorted if x % 2 != 0]
        if is_sum_free(cand_odd) and len(cand_odd) > best_size:
            best = list(cand_odd)
            best_size = len(cand_odd)
        
        if best is None:
            # Last resort: take one element
            best = [full_sorted[0]]
        
        # Try to improve by adding more elements
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each element not in best
            for x in full_sorted:
                if x in best_set:
                    continue
                # Check if adding x keeps sum-free
                ok = True
                for a in best:
                    if a + x in best_set:
                        ok = False
                        break
                    if a + x == x:  # a=0, impossible
                        ok = False
                        break
                if ok and x + x not in best_set:
                    best.append(x)
                    best_set.add(x)
                    improved = True
                    break
        
        return best

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < 2:
            # Not enough time, use quick fallback
            full_set = instance(size, seed)
            results.append([x for x in full_set if x % 2 != 0])
            continue
        
        # Allocate proportional time
        time_budget = min(remaining / (len(instances) - i), 30)
        deadline = start_time + TIME_LIMIT - 3
        
        full_set = instance(size, seed)
        result = solve_instance(full_set, deadline)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
