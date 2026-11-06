import time
import sys

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # We need a Sidon set S of size at least 9 (since C(8,2)=28 < 30)
    # Min diameter for n=9 is 44.
    # We search for n=9, D=44, 45, ...
    
    def is_sidon(s):
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
        
    def covers_1_to_30(s):
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                diffs.add(s[j] - s[i])
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
        
    def solve(n, D):
        # Backtracking search for Sidon set of size n in [0, D]
        # s[0] = 0
        s = [0]
        used_diffs = set()
        
        def backtrack(idx, last):
            if time.time() - start_time > deadline - 5:
                return None
            if idx == n:
                # Check if it covers 1..30
                if covers_1_to_30(s):
                    return s[:]
                return None
            
            # Heuristic: try values close to last? Or spread out?
            # To minimize max element, we are fixing D.
            # We need to pick s[idx] in (last, D].
            # Actually, we can pick any value > last.
            
            # Optimization: The largest element must be D?
            # Not necessarily, but if we fix D as the bound, we can try to include D.
            # If we don't include D, the max element is < D, which is better.
            # So we should search for max element < D first?
            # But we iterate D from 44 upwards.
            # So for a fixed D, we want to find ANY set in [0, D].
            
            # Pruning:
            # The next element x must satisfy:
            # 1. x - y not in used_diffs for all y in s
            # 2. x - y must not create a duplicate with existing diffs
            
            # Range of x: last + 1 to D
            # But we can prune.
            
            for x in range(last + 1, D + 1):
                valid = True
                new_diffs = []
                for y in s:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Tentatively add
                s.append(x)
                used_diffs.update(new_diffs)
                
                res = backtrack(idx + 1, x)
                if res is not None:
                    return res
                
                # Undo
                s.pop()
                used_diffs.difference_update(new_diffs)
                
            return None
            
        return backtrack(1, 0)

    for D in range(44, 100): # Upper bound 100 is safe
        res = solve(9, D)
        if res is not None:
            return res
            
    # Fallback: return a valid set found by greedy or larger D
    # If no set found for n=9, try n=10
    for D in range(54, 100):
        res = solve(10, D)
        if res is not None:
            return res
            
    # Final fallback: Greedy
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
