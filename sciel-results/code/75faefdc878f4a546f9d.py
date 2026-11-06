import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

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
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def conflict_out_degree(S):
        """Build conflict graph: node x has out-degree = number of (a,b) pairs in S where a+b=x or x+a=b or x+b=a."""
        S_set = set(S)
        S_list = list(S)
        n = len(S_list)
        out_deg = {x: 0 for x in S}
        # For each x, count how many pairs (a,b) in S create a conflict involving x
        # x conflicts if: x+a=b, x+b=a, or a+b=x for some a,b in S
        # Equivalently: for each x, count pairs (a,b) in S such that {x,a,b} forms a sum-free violation
        # a+b=x: for each a in S, if x-a in S, that's a conflict
        # x+a=b: for each a in S, if x+a in S, that's a conflict (same as b-a=x)
        # x+b=a: same as above
        for x in S_list:
            cnt = 0
            for a in S_list:
                if a == x:
                    # 2x = x only if x=0, but x!=0
                    continue
                # Check x + a = b (i.e., x+a in S)
                if x + a in S_set:
                    cnt += 1
                # Check a + b = x (i.e., x-a in S, and a != x-a to avoid double count? 
                # Actually a+b=x means b=x-a. If x-a in S, then (a, x-a) is a pair summing to x.
                # But we also count (x-a, a). To avoid double counting, only count if a <= x-a?
                # For out-degree, let's just count distinct pairs.
                # Let's count: for each a in S, if x-a in S and a != x-a, that's one unordered pair.
                # If a == x-a (i.e., 2a=x), that's one pair.
                if x - a in S_set:
                    if a < x - a:
                        cnt += 1
                    elif a == x - a:
                        cnt += 1
            out_deg[x] = cnt
        return out_deg

    def greedy_optimize(V, time_limit, rng):
        S = set(V)
        removed = list(V)
        
        # Phase 1: Iterative removal using conflict out-degree
        while not is_sum_free(S):
            if time.time() > start_time + time_limit:
                break
            
            out_deg = conflict_out_degree(S)
            if not out_deg:
                break
            
            max_deg = max(out_deg.values())
            if max_deg == 0:
                break
            
            candidates = [x for x, c in out_deg.items() if c == max_deg]
            x_to_remove = rng.choice(candidates)
            
            S.remove(x_to_remove)
            removed.remove(x_to_remove)

        # Phase 2: Local search with add-back and swap moves
        def can_add(x, S_set, S_list):
            """Check if x can be added to S without violating sum-free property."""
            for a in S_list:
                if x + a in S_set:
                    return False
            for a in S_list:
                if x - a in S_set:
                    return False
            return True

        def try_add_back(S, S_set, S_list, removed, rng):
            """Try to add back elements from removed. Returns list of added elements."""
            added = []
            new_removed = []
            for x in removed:
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_set.add(x)
                    S_list.append(x)
                    added.append(x)
                else:
                    new_removed.append(x)
            return added, new_removed

        def try_swap(S, S_set, S_list, removed, rng):
            """Try swap moves: remove one element from S, try to add two from removed."""
            improved = False
            S_list_copy = list(S_list)
            
            for x in S_list_copy:
                if time.time() > start_time + time_limit:
                    break
                # Try removing x and adding two elements from removed
                S.remove(x)
                S_set.remove(x)
                S_list.remove(x)
                
                # Try to add two elements from removed
                added = []
                new_removed = []
                for y in removed:
                    if can_add(y, S_set, S_list):
                        S.add(y)
                        S_set.add(y)
                        S_list.append(y)
                        added.append(y)
                    else:
                        new_removed.append(y)
                
                if len(added) >= 2:
                    # Net gain: removed 1, added >= 2
                    removed = new_removed
                    improved = True
                else:
                    # Revert
                    S.add(x)
                    S_set.add(x)
                    S_list.append(x)
                    for y in added:
                        S.remove(y)
                        S_set.remove(y)
                        S_list.remove(y)
                        removed.append(y)
                
                if improved:
                    break
            
            return improved

        # Multiple passes of add-back
        S_list = list(S)
        S_set = S
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            rng.shuffle(removed)
            S_list = list(S)
            S_set = S
            
            added, new_removed = try_add_back(S, S_set, S_list, removed, rng)
            if added:
                improved = True
                removed = new_removed
            
            # Try swap moves
            if time.time() < start_time + time_limit:
                if try_swap(S, S_set, list(S), removed, rng):
                    improved = True
        
        return list(S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_size = 0
        best_set = []
        
        # Time allocation per instance
        if size == 60:
            time_per_instance = 20.0
        elif size == 120:
            time_per_instance = 35.0
        else:
            time_per_instance = 55.0
        
        # Number of restarts
        restarts = 3
        time_per_restart = time_per_instance / restarts
        
        for r in range(restarts):
            if time.time() > start_time + 108:
                break
            
            rng = random.Random(hash((seed, r)) & 0xFFFFFFFF)
            res = greedy_optimize(V, time_per_restart, rng)
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
        
        # Use remaining time for additional restarts if available
        remaining = start_time + 108 - time.time()
        if remaining > 5:
            extra_restarts = int(remaining // 10)
            for r in range(restarts, restarts + extra_restarts):
                if time.time() > start_time + 108:
                    break
                rng = random.Random(hash((seed, r)) & 0xFFFFFFFF)
                res = greedy_optimize(V, min(10.0, remaining), rng)
                if len(res) > best_size:
                    best_size = len(res)
                    best_set = res
                remaining = start_time + 108 - time.time()
                if remaining <= 5:
                    break
        
        results.append(best_set)

    return results
