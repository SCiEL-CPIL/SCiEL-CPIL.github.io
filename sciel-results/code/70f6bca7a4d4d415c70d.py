import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def is_valid_set(S):
        n = len(S)
        if n < 3:
            return True
        
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
                        
            for k in range(2, m + 1):
                target = k * a
                if target > sum(others):
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def try_greedy(candidates):
        current = []
        for x in candidates:
            if len(current) >= 48:
                break
            if time.time() > deadline:
                break
            new_set = current + [x]
            if is_valid_set(new_set):
                current = new_set
        return current

    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= 48:
                    break
                new_set = S + [x]
                if is_valid_set(new_set):
                    S = new_set
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= 48:
                break
            
            # Try swapping
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = S_set - {removed}
                    for x in available:
                        if x in S_temp_set:
                            continue
                        new_set = S_temp + [x]
                        if is_valid_set(new_set):
                            S = new_set
                            S_set = S_temp_set | {x}
                            available.remove(x)
                            available.append(removed)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # 1. Structured Greedy Passes
    # Descending
    desc = list(range(512, 0, -1))
    res = try_greedy(desc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Ascending
    asc = list(range(1, 513))
    res = try_greedy(asc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Random Passes
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        res = try_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
        if best_size == 48:
            return best_set

    # 2. Local Search
    if best_set and time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == 48:
            return best_set

    # 3. More Local Search with restarts
    if best_set and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = list(best_set)
            rng = random.Random(time.time_ns())
            if perturbed:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
            improved = local_search(perturbed)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == 48:
                return best_set

    return best_set
