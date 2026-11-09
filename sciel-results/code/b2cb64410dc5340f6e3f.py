import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0
    
    target_diffs = set(range(1, 31))
    
    def check_time():
        return time.time() < DEADLINE

    def solve_for_n(n, M):
        S = [0]
        used_diffs = set()
        
        def dfs(S, used_diffs):
            if not check_time():
                return None
            
            if target_diffs.issubset(used_diffs):
                return S[:]
            
            if len(S) == n:
                return None
                
            missing = target_diffs - used_diffs
            s = len(S)
            k = n - s
            # Max possible new differences
            max_new = s * k + k * (k - 1) // 2
            if len(missing) > max_new:
                return None
            
            # Additional prune:
            # For each missing d, we need to be able to form it.
            # If d > M, impossible. (But M >= 30 and d <= 30)
            # If d is missing, check if it can be formed.
            # If for some d, no y in S allows y+d <= M, and we don't have enough
            # "slots" for new-new pairs, prune.
            # Simple version:
            # Count how many missing diffs CAN be formed with at least one existing element.
            # If a diff d can only be formed by new-new, it "costs" 2 new elements effectively?
            # No, just 1 new element if paired with existing.
            
            # Let's just use the count prune.
            
            last = S[-1]
            for x in range(last + 1, M + 1):
                valid = True
                new_diffs = set()
                for y in S:
                    d = x - y
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                result = dfs(S + [x], used_diffs | new_diffs)
                if result is not None:
                    return result
            return None
            
        return dfs(S, used_diffs)

    for M in range(30, 200):
        if not check_time():
            break
        for n in [9, 10, 11]:
            res = solve_for_n(n, M)
            if res:
                return res

    # Fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            if v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            trial = sorted(a + [x])
            d_list = diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                d_list = diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a
