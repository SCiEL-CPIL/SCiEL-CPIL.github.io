import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        for a in S_set:
            if (x - a) in S_set:
                return False
        for a in S_set:
            if (a + x) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def conflict_count(x, S_set):
        """Count how many conflicts x has with S_set (degree in hypergraph)."""
        count = 0
        for a in S_set:
            if (x - a) in S_set:
                count += 1
            if (a + x) in S_set:
                count += 1
        if (2 * x) in S_set:
            count += 1
        return count

    def greedy_build(S, order_key=None):
        """Build a sum-free set greedily, ordering elements by conflict_count if order_key provided."""
        S_set = set()
        remaining = list(S)
        if order_key is None:
            # Default: sort by absolute value descending (large magnitude first)
            remaining.sort(key=lambda x: -abs(x))
        else:
            remaining.sort(key=order_key)
        
        for x in remaining:
            if can_add(x, S_set):
                S_set.add(x)
        return list(S_set)

    def greedy_improve(S, time_limit):
        """Iteratively try to add elements from the complement."""
        S_set = set(S)
        all_set = set(S)
        remaining = [x for x in all_set if x not in S_set]
        
        start_time = time.time()
        while True:
            if time.time() - start_time > time_limit:
                break
            
            added_any = False
            # Sort remaining by conflict count (least conflicting first)
            remaining.sort(key=lambda x: conflict_count(x, S_set))
            
            for x in remaining[:]:
                if x in S_set:
                    continue
                if can_add(x, S_set):
                    S_set.add(x)
                    remaining.remove(x)
                    added_any = True
            
            if not added_any:
                break
        
        return list(S_set)

    def local_search_swaps(S, time_limit):
        """Try to improve by swapping: remove one element, try to add one or more others."""
        S_set = set(S)
        all_elements = set(S)
        complement = [x for x in all_elements if x not in S_set]
        
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try removing each element in S_set
            for remove_candidate in list(S_set):
                if time.time() - start_time > time_limit:
                    break
                S_set.remove(remove_candidate)
                # Try to add as many from complement + removed as possible
                candidates = complement + [remove_candidate]
                candidates.sort(key=lambda x: conflict_count(x, S_set))
                added = 0
                for x in candidates:
                    if x not in S_set and can_add(x, S_set):
                        S_set.add(x)
                        added += 1
                if added > 1:
                    improved = True
                else:
                    # Revert if no improvement
                    S_set.add(remove_candidate)
        
        return list(S_set)

    def get_suffix(S):
        n = len(S)
        best_k = 1
        for k in range(1, n + 1):
            if 2 * S[n - k] > S[n - 1]:
                best_k = k
            else:
                break
        return S[n - best_k : n]

    def solve_instance(S, time_limit):
        """Solve a single instance with multiple strategies and local search."""
        start_time = time.time()
        
        # Strategy 1: Odds
        odds = [x for x in S if x % 2 != 0]
        
        # Strategy 2: Suffix
        suffix = get_suffix(S)
        
        # Strategy 3: Greedy by descending abs value
        greedy_abs = greedy_build(S, order_key=lambda x: -abs(x))
        
        # Strategy 4: Greedy by ascending conflict (least conflicting first)
        greedy_conflict = greedy_build(S, order_key=lambda x: conflict_count(x, set(S)))
        
        # Strategy 5: Greedy by ascending value
        greedy_asc = greedy_build(S, order_key=lambda x: x)
        
        # Strategy 6: Greedy by descending value
        greedy_desc = greedy_build(S, order_key=lambda x: -x)
        
        candidates = [odds, suffix, greedy_abs, greedy_conflict, greedy_asc, greedy_desc]
        best_init = max(candidates, key=len)
        
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 1
        
        if remaining_time <= 0:
            return best_init
        
        # Improve the best initial
        improved = greedy_improve(best_init, remaining_time)
        
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 1
        
        if remaining_time <= 0:
            return improved
        
        # Local search with swaps
        polished = local_search_swaps(improved, remaining_time)
        
        # Verify and return
        if is_sum_free(polished):
            return polished
        else:
            return improved

    # Instances
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    total_time = 145  # Safety margin before 150s kill
    start = time.time()
    
    for i, S in enumerate(instances):
        elapsed = time.time() - start
        # Allocate more time to larger instances
        if i == 0:
            time_limit = 30
        elif i == 1:
            time_limit = 45
        else:
            time_limit = 60
        
        remaining_time = total_time - elapsed
        if remaining_time <= 0:
            results.append([x for x in S if x % 2 != 0])
            continue
        
        time_limit = min(time_limit, remaining_time - 5)  # 5s safety margin
        
        if time_limit <= 0:
            results.append([x for x in S if x % 2 != 0])
            continue
        
        res = solve_instance(S, time_limit)
        results.append(res)
    
    # Final verification
    for i, res in enumerate(results):
        if not is_sum_free(res):
            # Fallback to odds
            results[i] = [x for x in instances[i] if x % 2 != 0]
    
    return results
