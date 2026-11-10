import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    def compute_sums(chosen):
        n = len(chosen)
        S = [set() for _ in range(n + 1)]
        S[0].add(0)
        for x in chosen:
            for k in range(n, 0, -1):
                if k - 1 >= 0 and k - 1 < n:
                    S[k] = S[k] | {s + x for s in S[k-1]}
        return S

    def count_conflicts(S):
        n = len(S) - 1
        total = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                total += len(S[i] & S[j])
        return total

    def local_search(initial_set):
        chosen = list(initial_set)
        S = compute_sums(chosen)
        current_conflicts = count_conflicts(S)
        
        removed_pool = []
        
        # Phase 1: Remove elements to reduce conflicts
        while current_conflicts > 0 and len(chosen) > 1:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_conflicts = current_conflicts
            best_new_S = None
            
            n = len(chosen)
            indices = list(range(n))
            random.shuffle(indices)
            
            for idx in indices:
                new_chosen = chosen[:idx] + chosen[idx+1:]
                new_S = compute_sums(new_chosen)
                new_conflicts = count_conflicts(new_S)
                
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_idx = idx
                    best_new_S = new_S
                    
                    if new_conflicts == 0:
                        break
            
            if best_idx != -1:
                removed_pool.append(chosen[best_idx])
                chosen = chosen[:best_idx] + chosen[best_idx+1:]
                S = best_new_S
                current_conflicts = best_conflicts
            else:
                break
        
        # Phase 2: Try to add elements back or new ones
        if current_conflicts == 0:
            candidates = list(removed_pool)
            all_nums = set(range(1, 257))
            chosen_set = set(chosen)
            for num in all_nums:
                if num not in chosen_set:
                    candidates.append(num)
            
            random.shuffle(candidates)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for c in candidates[:]:
                    if time.time() > deadline:
                        break
                    if c in chosen_set:
                        continue
                    
                    new_chosen = chosen + [c]
                    new_S = compute_sums(new_chosen)
                    new_conflicts = count_conflicts(new_S)
                    
                    if new_conflicts == 0:
                        chosen = new_chosen
                        S = new_S
                        chosen_set.add(c)
                        if c in candidates:
                            candidates.remove(c)
                        improved = True
                        break
        
        return chosen

    best_result = []
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        init_size = 40
        if attempt == 1: init_size = 50
        if attempt == 2: init_size = 30
        
        initial_set = random.sample(range(1, 257), init_size)
        
        result = local_search(initial_set)
        
        if len(result) > len(best_result):
            best_result = result
            
    if len(best_result) < 10:
        chosen = []
        for x in range(1, 257):
            new_chosen = chosen + [x]
            S = compute_sums(new_chosen)
            if count_conflicts(S) == 0:
                chosen = new_chosen
        best_result = chosen

    return [int(x) for x in best_result]
