import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    target_diffs = set(range(1, 31))
    
    def solve(M, k):
        S = [0]
        used_diffs = set()
        covered = set()
        
        def backtrack(idx, last_x):
            if time.time() > deadline:
                return None
            if idx == k:
                if target_diffs.issubset(covered):
                    return S.copy()
                return None
            
            rem = k - idx
            max_new = rem * idx + rem * (rem - 1) // 2
            if len(target_diffs - covered) > max_new:
                return None
            
            candidates = []
            for x in range(last_x + 1, M + 1):
                valid = True
                new_diffs = []
                for y in S:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_covered = 0
                for d in new_diffs:
                    if d in target_diffs and d not in covered:
                        new_covered += 1
                
                candidates.append((x, new_covered, new_diffs))
            
            candidates.sort(key=lambda t: t[1], reverse=True)
            
            for x, nc, new_diffs in candidates:
                S.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                    if d in target_diffs:
                        covered.add(d)
                
                res = backtrack(idx + 1, x)
                if res is not None:
                    return res
                
                S.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                    if d in target_diffs:
                        covered.discard(d)
                        
            return None
            
        return backtrack(1, 0)

    for M in range(30, 101):
        for k in range(9, 13):
            if time.time() > deadline:
                break
            res = solve(M, k)
            if res is not None:
                return res
        
    # Fallback greedy
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
