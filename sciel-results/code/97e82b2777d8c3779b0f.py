import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s margin

    def seed_solution():
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    best_sol = None
    best_val = float('inf')
    
    # Try to find a solution with smaller diameter than seed
    # We don't strictly need to compute seed first, but it's a good fallback.
    # Let's just run the search.
    
    for k in [9, 10, 11]:
        for M in range(30, 150):
            if time.time() - start_time > time_limit:
                break
            
            sol = search_sidon(M, k, start_time, time_limit)
            if sol is not None:
                max_val = max(sol)
                if max_val < best_val:
                    best_val = max_val
                    best_sol = sol
                break # Found min M for this k
        
        if time.time() - start_time > time_limit:
            break
        
        if best_sol is not None and k == 9:
            break

    if best_sol is not None:
        return best_sol
    
    return seed_solution()

def search_sidon(M, k, start_time, time_limit):
    target_diffs = set(range(1, 31))
    current_set = [0]
    used_diffs = set()
    missing_diffs = set(target_diffs)
    
    def backtrack():
        if time.time() - start_time > time_limit:
            return "TIMEOUT"
            
        if not missing_diffs:
            return list(current_set)
        
        if len(current_set) == k:
            return None
        
        candidates = []
        # Optimization: Only consider x that covers at least one missing diff
        for x in range(1, M + 1):
            if x in current_set:
                continue
            
            valid = True
            covered = 0
            new_diffs = []
            for y in current_set:
                d = abs(x - y)
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                if d in missing_diffs:
                    covered += 1
            
            if not valid:
                continue
            
            if covered == 0:
                continue
                
            candidates.append((covered, x))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda t: -t[0])
        
        for _, x in candidates:
            current_set.append(x)
            
            old_missing = set(missing_diffs)
            new_used = []
            for y in current_set[:-1]:
                d = abs(x - y)
                used_diffs.add(d)
                new_used.append(d)
                if d in missing_diffs:
                    missing_diffs.remove(d)
            
            res = backtrack()
            if res == "TIMEOUT":
                return "TIMEOUT"
            if res is not None:
                return res
            
            current_set.pop()
            for d in new_used:
                used_diffs.remove(d)
            missing_diffs = old_missing
            
        return None

    res = backtrack()
    if res == "TIMEOUT":
        return None
    return res
