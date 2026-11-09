import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    target_diffs = set(range(1, 31))
    
    # Try increasing diameters
    # Lower bound: M must be at least 30.
    # Also k(k-1)/2 <= M.
    # Min k is 9.
    # So M >= 36 for k=9.
    
    for M in range(36, 200):
        if time.time() > deadline:
            break
            
        # Max k for this M
        max_k = 1
        while max_k * (max_k - 1) // 2 <= M:
            max_k += 1
        max_k -= 1
        
        # Min k is 9
        min_k = 9
        
        if max_k < min_k:
            continue
            
        for k in range(min_k, max_k + 1):
            if time.time() > deadline:
                break
                
            solution = search_sidon(M, k, target_diffs, deadline)
            if solution:
                return solution
                
    # Fallback
    return greedy_construct()

def search_sidon(M, k, target_diffs, deadline):
    current_set = [0]
    current_diffs = set()
    
    def backtrack(s, diffs, missing):
        if time.time() > deadline:
            return None
        
        if not missing:
            return s[:]
            
        if len(s) == k:
            return None
            
        candidates = set()
        for y in s:
            for d in missing:
                x1 = y + d
                if 0 <= x1 <= M and x1 not in s:
                    candidates.add(x1)
                x2 = y - d
                if 0 <= x2 <= M and x2 not in s:
                    candidates.add(x2)
        
        # Try candidates
        # Sort to ensure we explore smaller x first? 
        # Doesn't matter much for existence, but helps find "tight" sets.
        for x in sorted(candidates):
            new_diffs = set()
            valid = True
            for y in s:
                d = abs(x - y)
                if d in diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            if not (new_diffs & missing):
                continue
                
            new_missing = missing - new_diffs
            new_s = s + [x]
            new_s.sort()
            new_diffs_set = diffs | new_diffs
            
            res = backtrack(new_s, new_diffs_set, new_missing)
            if res:
                return res
                
        return None

    return backtrack([0], set(), set(target_diffs))

def greedy_construct():
    a = [0]
    def get_diffs(s):
        return set(y - x for i, x in enumerate(s) for y in s[i + 1:])
    
    def is_sidon(s):
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        return len(diffs) == len(set(diffs))
        
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        for x in range(1, 1000):
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a
