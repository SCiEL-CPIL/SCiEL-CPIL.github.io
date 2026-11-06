import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0
    
    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')
    
    def dfs(S, used_diffs):
        nonlocal best_solution, best_max_val, start_time, TIME_LIMIT
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if target_diffs.issubset(used_diffs):
            current_max = max(S)
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = list(S)
                if current_max == 30:
                    return
            return

        if max(S) >= best_max_val:
            return
            
        if len(S) > 32:
            return
            
        missing = target_diffs - used_diffs
        if not missing:
            return

        candidates = []
        # Look ahead a bit
        # Upper bound for x: best_max_val - 1
        # Also, we don't need to go too far if we have a good best_max_val
        upper_bound = min(best_max_val - 1, max(S) + 50) 
        # 50 is a heuristic window. If optimal is tight, it's in here.
        
        for x in range(max(S) + 1, upper_bound + 1):
            # Check Sidon
            valid = True
            new_diffs = []
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Calculate coverage
            coverage = 0
            for d in new_diffs:
                if d in missing:
                    coverage += 1
            
            candidates.append((x, coverage, new_diffs))
            
        # Sort by x first, then by coverage descending
        candidates.sort(key=lambda item: (item[0], -item[1]))
        
        for x, coverage, new_diffs in candidates:
            new_S = S + [x]
            new_used_diffs = used_diffs.union(new_diffs)
            dfs(new_S, new_used_diffs)
            
            if best_max_val == 30:
                return

    S_init = [0, 1]
    used_diffs_init = {1}
    
    dfs(S_init, used_diffs_init)
    
    if best_solution is None:
        # Fallback
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
        best_solution = a

    return best_solution
