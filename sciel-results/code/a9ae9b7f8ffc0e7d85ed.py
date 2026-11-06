import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    req_diffs_mask = 0
    for d in range(1, 31):
        req_diffs_mask |= (1 << d)
        
    def check_time():
        return time.time() > deadline

    def dfs(current_set, used_diffs_mask, remaining_reqs, waste_count, M, k_target):
        if check_time():
            return None
            
        if not remaining_reqs:
            return list(current_set)
            
        if len(current_set) >= k_target:
            return None
            
        # Prune: if waste_count > allowed_waste, return None
        allowed_waste = (k_target * (k_target - 1) // 2) - 30
        if waste_count > allowed_waste:
            return None
            
        last = current_set[-1]
        
        # Generate candidates
        # We want to prioritize x that cover many remaining_reqs
        # But iterating all x is costly.
        
        # Let's try to find x such that it covers at least one remaining_req
        # Or just iterate and prune.
        
        # To speed up, we can precompute valid x?
        # No, state changes.
        
        # Order candidates by "coverage"
        candidates = []
        for x in range(last + 1, M + 1):
            valid = True
            new_diffs = 0
            new_waste = 0
            for a in current_set:
                d = x - a
                bit = 1 << d
                if used_diffs_mask & bit:
                    valid = False
                    break
                new_diffs |= bit
                if not (remaining_reqs & bit):
                    new_waste += 1
            
            if valid:
                # Count how many reqs this x covers
                covered = remaining_reqs & new_diffs
                # Heuristic: prioritize x with high coverage
                candidates.append((bin(covered).count('1'), x, new_diffs, new_waste))
                
        # Sort by coverage descending
        candidates.sort(reverse=True)
        
        for _, x, new_diffs, new_waste in candidates:
            new_mask = used_diffs_mask | new_diffs
            new_remaining = remaining_reqs & ~new_diffs
            new_waste_count = waste_count + new_waste
            
            res = dfs(current_set + [x], new_mask, new_remaining, new_waste_count, M, k_target)
            if res is not None:
                return res
                
        return None

    for M in range(30, 100):
        if check_time():
            break
            
        initial_set = [0]
        initial_mask = 0
        initial_remaining = req_diffs_mask
        
        # Try k=9
        res = dfs(initial_set, initial_mask, initial_remaining, 0, M, 9)
        if res is not None:
            return res
            
        # Try k=10
        res = dfs(initial_set, initial_mask, initial_remaining, 0, M, 10)
        if res is not None:
            return res

    # Fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
