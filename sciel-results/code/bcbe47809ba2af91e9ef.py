import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # 5 seconds margin
    
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
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                if a + b in S_set:
                    return False
        return True
    
    def count_violations(S):
        S_set = set(S)
        n = len(S)
        count = 0
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                if a + b in S_set:
                    count += 1
        return count
    
    def greedy_remove(S, tie_breaker='abs_desc'):
        C = list(S)
        C_set = set(C)
        
        while True:
            if time.time() > deadline:
                break
                
            conflicts = {x: 0 for x in C}
            found_violation = False
            
            n = len(C)
            for i in range(n):
                a = C[i]
                for j in range(i, n):
                    b = C[j]
                    s = a + b
                    if s in C_set:
                        found_violation = True
                        conflicts[a] += 1
                        conflicts[b] += 1
                        conflicts[s] += 1
                        
            if not found_violation:
                break
            
            max_conf = max(conflicts.values())
            candidates = [x for x, c in conflicts.items() if c == max_conf]
            
            if len(candidates) == 1:
                to_remove = candidates[0]
            else:
                if tie_breaker == 'abs_desc':
                    to_remove = max(candidates, key=lambda x: abs(x))
                elif tie_breaker == 'abs_asc':
                    to_remove = min(candidates, key=lambda x: abs(x))
                elif tie_breaker == 'val_desc':
                    to_remove = max(candidates, key=lambda x: x)
                else:
                    to_remove = min(candidates, key=lambda x: x)
                    
            C.remove(to_remove)
            C_set.remove(to_remove)
            
        return C

    def local_search_swap(S, max_iter=100):
        """Try to improve a sum-free set by removing one element and adding another."""
        if not is_sum_free(S):
            return S
        
        C = list(S)
        C_set = set(C)
        S_set = set(S)
        excluded = S_set
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try removing each element and see if we can add a better one
            for i in range(len(C)):
                if time.time() > deadline:
                    break
                removed = C[i]
                C_set.remove(removed)
                C.pop(i)
                
                # Now try to add elements from S that aren't in C
                best_add = None
                best_score = -1
                for x in S:
                    if x in C_set:
                        continue
                    # Check if adding x keeps sum-free
                    valid = True
                    if 2*x in C_set:
                        valid = False
                    else:
                        for c in C:
                            if x + c in C_set or c - x in C_set:
                                valid = False
                                break
                    if valid:
                        # Score: prefer elements that don't create potential conflicts
                        # Just add it and count how many new conflicts would be created
                        # Actually, if it's valid, it creates 0 conflicts
                        # So we just want to add any valid element
                        best_add = x
                        break
                
                if best_add is not None:
                    C.append(best_add)
                    C_set.add(best_add)
                    improved = True
                    # Reset i since we modified C
                    i = -1
                else:
                    # Put removed back
                    C.insert(i, removed)
                    C_set.add(removed)
        
        return C

    def odd_subset(S):
        return [x for x in S if x % 2 != 0]

    def greedy_add(S, order_key, deadline_check=True):
        C = []
        C_set = set()
        for x in sorted(S, key=order_key):
            if deadline_check and time.time() > deadline:
                break
            valid = True
            # Check if x is sum of two in C
            for i in range(len(C)):
                a = C[i]
                for j in range(i, len(C)):
                    b = C[j]
                    if a + b == x:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                if 2*x in C_set:
                    valid = False
                else:
                    for c in C:
                        if x + c in C_set:
                            valid = False
                            break
                    if valid:
                        for c in C:
                            if c - x in C_set:
                                valid = False
                                break
            
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def randomized_greedy_remove(S, num_restarts=5):
        """Greedy removal with random tie-breaking."""
        best = []
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            C = list(S)
            C_set = set(C)
            
            while True:
                if time.time() > deadline:
                    break
                    
                conflicts = {x: 0 for x in C}
                found_violation = False
                
                n = len(C)
                for i in range(n):
                    a = C[i]
                    for j in range(i, n):
                        b = C[j]
                        s = a + b
                        if s in C_set:
                            found_violation = True
                            conflicts[a] += 1
                            conflicts[b] += 1
                            conflicts[s] += 1
                            
                if not found_violation:
                    break
                
                max_conf = max(conflicts.values())
                candidates = [x for x, c in conflicts.items() if c == max_conf]
                to_remove = random.choice(candidates)
                
                C.remove(to_remove)
                C_set.remove(to_remove)
            
            if len(C) > len(best):
                best = C
        
        return best

    results = []
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    for S in instances:
        best_set = []
        
        # Baseline: Odds
        odds = odd_subset(S)
        if len(odds) > len(best_set):
            best_set = odds
            
        # Try greedy removal with different tie-breakers
        for tb in ['abs_desc', 'abs_asc', 'val_desc', 'val_asc']:
            if time.time() > deadline:
                break
            res = greedy_remove(S, tb)
            if len(res) > len(best_set):
                best_set = res
        
        # Randomized greedy removal
        if time.time() <= deadline:
            res_rand = randomized_greedy_remove(S, num_restarts=10)
            if len(res_rand) > len(best_set):
                best_set = res_rand
        
        # Greedy Add strategies
        if time.time() <= deadline:
            res_add = greedy_add(S, key=lambda x: abs(x))
            if len(res_add) > len(best_set):
                best_set = res_add
                
            res_add2 = greedy_add(S, key=lambda x: -abs(x))
            if len(res_add2) > len(best_set):
                best_set = res_add2
        
        # Local search swap on best set found
        if time.time() <= deadline and len(best_set) > 0:
            res_ls = local_search_swap(best_set, max_iter=50)
            if len(res_ls) > len(best_set):
                best_set = res_ls
        
        results.append(best_set)
        
    return results
