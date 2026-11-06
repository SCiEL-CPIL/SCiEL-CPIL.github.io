import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_set, candidate):
        """Check if candidate can be added to S_set to keep it sum-free."""
        if candidate in S_set:
            return False
        for x in S_set:
            if (candidate + x) in S_set:
                return False
            if (candidate - x) in S_set:
                return False
        return True

    def greedy_add(S_list, candidates, S_set=None):
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        for c in candidates:
            if can_add(S_set, c):
                result.append(c)
                S_set.add(c)
        return result, S_set

    def repair(S_list):
        """Remove elements until sum-free."""
        if not S_list:
            return []
        S_set = set(S_list)
        result = list(S_list)
        changed = True
        while changed and result:
            changed = False
            i = 0
            while i < len(result):
                x = result[i]
                conflict = False
                for y in S_set:
                    if y == x:
                        continue
                    if (x + y) in S_set:
                        conflict = True
                        break
                if conflict:
                    result.pop(i)
                    S_set.remove(x)
                    changed = True
                else:
                    i += 1
        return result

    def local_search(S_list, values, time_limit=5.0):
        """Improve S_list via swaps and additions."""
        S_set = set(S_list)
        values_set = set(values)
        start = time.time()
        best_list = list(S_list)
        best_size = len(best_list)
        
        # Phase 1: Greedy add missing elements
        missing = [v for v in values if v not in S_set]
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)
        
        # Phase 2: Swap-based local search
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            current_size = len(S_list)
            
            for i, x in enumerate(list(S_list)):
                if (time.time() - start) >= time_limit:
                    break
                
                # Try removing x
                S_test_set = S_set - {x}
                
                # Try adding each missing element
                for y in values:
                    if y in S_test_set:
                        continue
                    # Check if y can replace x
                    if can_add(S_test_set, y):
                        # Accept the swap
                        S_list[i] = y
                        S_set = S_test_set
                        S_set.add(y)
                        improved = True
                        break
                
                if improved:
                    break
        
        # Final greedy add
        missing = [v for v in values if v not in S_set]
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)
        
        return S_list

    def solve_one(values, time_budget=30.0):
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        max_v = max(values)
        min_v = min(values)
        
        # Generate initial candidates
        candidates = []
        
        # Odds
        candidates.append([x for x in values if x % 2 != 0])
        
        # Large positives
        thresh_pos = max_v / 2.0
        candidates.append([x for x in values if x > thresh_pos])
        
        # Large negatives
        thresh_neg = min_v / 2.0
        candidates.append([x for x in values if x < thresh_neg])
        
        # Union of large pos and neg
        s4 = sorted(set([x for x in values if x > thresh_pos or x < thresh_neg]))
        candidates.append(s4)
        
        # Greedy by absolute value
        s5 = []
        s5_set = set()
        for x in sorted(values, key=abs, reverse=True):
            if can_add(s5_set, x):
                s5.append(x)
                s5_set.add(x)
        candidates.append(s5)
        
        # Greedy by value
        s6 = []
        s6_set = set()
        for x in sorted(values):
            if can_add(s6_set, x):
                s6.append(x)
                s6_set.add(x)
        candidates.append(s6)
        
        # Random greedy
        rng = random.Random(42)
        shuffled = list(values)
        rng.shuffle(shuffled)
        s7, _ = greedy_add([], shuffled)
        candidates.append(s7)
        
        # Find best initial candidate
        best = []
        for c in candidates:
            c_valid = repair(c)
            if len(c_valid) > len(best):
                best = c_valid
        
        # Local search on best
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining > 2:
            improved = local_search(best, values, time_limit=min(remaining - 1, 10.0))
            if len(improved) > len(best):
                best = improved
        
        # Random restarts if time permits
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        num_restarts = max(0, int(remaining / 5.0))
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            shuffled = list(values)
            rng.shuffle(shuffled)
            trial, _ = greedy_add([], shuffled)
            
            if len(trial) >= len(best) - 2:
                # Only do local search if promising
                ts = local_search(trial, values, time_limit=3.0)
                if len(ts) > len(best):
                    best = ts
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            results.append([])
            continue
        
        values = instance(size, seed)
        
        # Allocate time based on instance size
        if size <= 60:
            tb = 25.0
        elif size <= 120:
            tb = 35.0
        else:
            tb = 45.0
        
        subset = solve_one(values, time_budget=tb)
        
        # Final verification and fallback
        if not is_sum_free(subset):
            subset = repair(subset)
        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]
        
        results.append(subset)
    
    return results
