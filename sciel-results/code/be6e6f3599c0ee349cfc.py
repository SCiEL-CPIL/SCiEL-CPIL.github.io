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

    def walksat_search(S, time_limit, rng):
        """
        WalkSAT-style local search for sum-free subset.
        Variables: inclusion of each element.
        Clauses: for each triple (a, b, a+b) where a+b in S, at most 2 can be included.
        We maximize number of included variables while satisfying all clauses.
        
        Strategy: Start from a valid solution, then try flips (add/remove) that
        maintain validity or temporarily violate and repair.
        """
        S_set = set(S)
        all_elements = set(S)
        complement = [x for x in all_elements if x not in S_set]
        
        # Precompute all conflict triples for fast clause evaluation
        # For each x, find all pairs (a, b) in S such that a + b = x or x + a = b or x + b = a
        # Actually, we need: for each x, the set of y in S such that x + y in S or x - y in S or y - x in S
        
        def get_conflicts(x, current_set):
            """Get all elements in current_set that conflict with x."""
            conflicts = set()
            for y in current_set:
                if x + y in current_set:
                    conflicts.add(y)
                if x - y in current_set:
                    conflicts.add(y)
                if y - x in current_set:
                    conflicts.add(y)
            return conflicts
        
        def num_violations(current_set):
            """Count number of violated clauses (triples with all 3 present)."""
            count = 0
            for a in current_set:
                for b in current_set:
                    if a + b in current_set and a + b != a and a + b != b:
                        # Avoid double counting: only count if a <= b < a+b or similar
                        # Actually, let's just count each unique triple once
                        if a <= b:
                            count += 1
            return count

        def try_flip_add(x, current_set):
            """Try to add x to current_set. Returns (success, new_set)."""
            if x in current_set:
                return False, current_set
            # Check if adding x creates any conflicts
            for y in current_set:
                if x + y in current_set:
                    return False, current_set
                if x - y in current_set:
                    return False, current_set
                if y - x in current_set:
                    return False, current_set
            return True, current_set | {x}

        def try_flip_remove(x, current_set):
            """Try to remove x from current_set. Returns (success, new_set)."""
            if x not in current_set:
                return False, current_set
            return True, current_set - {x}

        best_set = set(S_set)
        best_size = len(S_set)
        
        start_time = time.time()
        restart_count = 0
        max_restarts = 10000
        
        while time.time() - start_time < time_limit and restart_count < max_restarts:
            restart_count += 1
            
            # Start from best solution found so far
            current_set = set(best_set)
            
            # WalkSAT steps
            for step in range(500):
                if time.time() - start_time > time_limit:
                    break
                
                # Try to add a random element from complement
                if complement:
                    x = rng.choice(complement)
                    if x not in current_set:
                        success, new_set = try_flip_add(x, current_set)
                        if success:
                            current_set = new_set
                            if len(current_set) > best_size:
                                best_set = set(current_set)
                                best_size = len(current_set)
                            continue
                
                # Try to remove a random element and see if we can add more
                if current_set:
                    x = rng.choice(list(current_set))
                    success, new_set = try_flip_remove(x, current_set)
                    if success:
                        # Try to greedily add back as many as possible
                        temp_set = set(new_set)
                        candidates = [y for y in all_elements if y not in temp_set]
                        candidates.sort(key=lambda y: conflict_count(y, temp_set))
                        for y in candidates:
                            if y not in temp_set and can_add(y, temp_set):
                                temp_set.add(y)
                        if len(temp_set) > len(current_set):
                            current_set = temp_set
                            if len(current_set) > best_size:
                                best_set = set(current_set)
                                best_size = len(current_set)
            
            # Periodically try more aggressive moves: remove 2 elements, add 3
            if restart_count % 10 == 0 and len(current_set) >= 2:
                if time.time() - start_time > time_limit:
                    break
                elems = list(current_set)
                r1 = rng.choice(elems)
                elems.remove(r1)
                r2 = rng.choice(elems)
                
                temp_set = current_set - {r1, r2}
                candidates = [y for y in all_elements if y not in temp_set]
                candidates.sort(key=lambda y: conflict_count(y, temp_set))
                for y in candidates:
                    if y not in temp_set and can_add(y, temp_set):
                        temp_set.add(y)
                if len(temp_set) > len(current_set):
                    current_set = temp_set
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
        
        return list(best_set)

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
        improved = greedy_improve(best_init, remaining_time * 0.3)
        
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 1
        
        if remaining_time <= 0:
            return improved
        
        # Local search with swaps
        polished = local_search_swaps(improved, remaining_time * 0.3)
        
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 1
        
        if remaining_time <= 0:
            return polished if is_sum_free(polished) else improved
        
        # WalkSAT-style search
        rng = random.Random(hash(tuple(S)) % (2**32))
        walksat_result = walksat_search(polished, remaining_time * 0.4, rng)
        
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 1
        
        # Final greedy improvement on walksat result
        if is_sum_free(walksat_result):
            final = greedy_improve(walksat_result, remaining_time * 0.3)
            if is_sum_free(final):
                return final
        
        # Return best valid solution
        if is_sum_free(polished):
            return polished
        elif is_sum_free(improved):
            return improved
        else:
            return best_init

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
