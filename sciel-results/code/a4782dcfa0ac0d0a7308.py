import random
import time

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    def get_diffs_set(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
    
    def is_sidon(S):
        n = len(S)
        if n < 2:
            return True
        diffs = get_diffs_set(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_all_diffs(diffs):
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    best_set = None
    best_max = float('inf')
    
    target = set(range(1, 31))
    
    for restart in range(2000):
        if time.time() - start_time > deadline:
            break
        
        S = [0]
        diffs = set()
        
        max_iter = 40
        for _ in range(max_iter):
            if diffs.issuperset(target):
                break
            
            missing = target - diffs
            
            candidates = set()
            for d in missing:
                for x in S:
                    c1 = x + d
                    if c1 >= 0 and c1 not in S:
                        candidates.add(c1)
                    c2 = x - d
                    if c2 >= 0 and c2 not in S:
                        candidates.add(c2)
            
            valid_candidates = []
            for c in candidates:
                new_S = sorted(S + [c])
                if is_sidon(new_S):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                break
            
            best_c = None
            best_cover_count = -1
            for c in valid_candidates:
                new_S = sorted(S + [c])
                new_diffs = get_diffs_set(new_S)
                new_missing = target - new_diffs
                cover_count = len(target) - len(new_missing)
                if cover_count > best_cover_count:
                    best_cover_count = cover_count
                    best_c = c
            
            if best_c is None:
                best_c = min(valid_candidates)
            
            S.append(best_c)
            S.sort()
            diffs = get_diffs_set(S)
        
        if covers_all_diffs(diffs):
            max_elem = max(S)
            if max_elem < best_max:
                best_max = max_elem
                best_set = S[:]
                
                improved = True
                while improved:
                    improved = False
                    for i in range(len(best_set)):
                        orig = best_set[i]
                        if orig == 0:
                            continue
                        for new_val in range(orig - 1, 0, -1):
                            if new_val in best_set:
                                continue
                            new_S = sorted([new_val if x == orig else x for x in best_set])
                            if is_sidon(new_S) and covers_all_diffs(get_diffs_set(new_S)):
                                if max(new_S) < max(best_set):
                                    best_set = new_S
                                    best_max = max(new_S)
                                    improved = True
                                    break
    
    if best_set:
        return best_set
    else:
        a = [0]
        def diffs_list(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in set(diffs_list(sorted(a))):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                        a = trial
                        break
                    y += 1
        return a
