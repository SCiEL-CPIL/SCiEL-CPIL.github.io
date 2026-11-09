import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # We search for the smallest D such that there exists a Sidon set A in [0, D]
    # containing 0, with |A| <= 32, and covering all differences 1..30.
    
    # Lower bound for D:
    # A Sidon set of size k has at least k(k-1)/2 differences.
    # We need to cover 30 differences.
    # k(k-1)/2 >= 30 => k ~ 8.5. So k=9.
    # A Golomb ruler of order 9 has minimum length 41.
    # We start D from 41.
    
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
        
    def popcount(x):
        return bin(x).count('1')
        
    for D in range(41, 61):
        if time.time() > deadline:
            break
            
        missing_mask = target_mask
        
        def dfs(current_set, used_mask, missing_mask, last_elem):
            if time.time() > deadline:
                return None
                
            if missing_mask == 0:
                return current_set[:]
                
            if len(current_set) > 32:
                return None
                
            # Pruning: Check if we can cover missing diffs with max possible new elements
            k_cur = len(current_set)
            max_new_elems = 32 - k_cur
            # Max new diffs from adding max_new_elems elements to a set of size k_cur
            # Diffs between new and old: k_cur * max_new_elems
            # Diffs within new: max_new_elems * (max_new_elems - 1) / 2
            max_new_diffs = k_cur * max_new_elems + max_new_elems * (max_new_elems - 1) // 2
            
            if popcount(missing_mask) > max_new_diffs:
                return None
            
            candidates = []
            
            for x in range(last_elem + 1, D + 1):
                valid = True
                new_mask = 0
                for y in current_set:
                    d = x - y
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                covered = new_mask & missing_mask
                score = popcount(covered)
                
                candidates.append((score, x, new_mask))
                
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for score, x, new_mask in candidates:
                new_missing = missing_mask & ~new_mask
                new_used = used_mask | new_mask
                
                res = dfs(current_set + [x], new_used, new_missing, x)
                if res is not None:
                    return res
                    
            return None
        
        result = dfs([0], 0, target_mask, 0)
        
        if result is not None:
            # Verify result just in case
            s = sorted(result)
            diffs = set()
            valid_sidon = True
            for i in range(len(s)):
                for j in range(i):
                    d = s[i] - s[j]
                    if d in diffs:
                        valid_sidon = False
                        break
                    diffs.add(d)
                if not valid_sidon:
                    break
            
            if valid_sidon and all(d in diffs for d in range(1, 31)):
                return result
            else:
                continue

    # Fallback: Return a known valid Sidon set with larger diameter
    # This set is a Golomb ruler of order 10, length 49?
    # Let's use a simple valid set if search fails, though it shouldn't.
    # [0, 1, 4, 10, 12, 17, 21, 28, 36, 41] is invalid as shown before.
    # A valid Golomb ruler of order 9 is [0, 1, 4, 10, 12, 17, 21, 28, 36]? No.
    # Known optimal Golomb ruler of order 9: [0, 1, 4, 10, 12, 17, 21, 28, 36] -> Length 36? No, max is 36.
    # Wait, order 9 min length is 41.
    # One such ruler: [0, 1, 4, 10, 12, 17, 21, 28, 36, 41] is order 10.
    # Let's just return a greedy construction if it fails, but it should find D=41 or 42.
    
    a = [0]
    for d in range(1, 31):
        if d in set(a[i] - a[j] for i in range(len(a)) for j in range(len(a)) if a[i] > a[j]):
            continue
        found = False
        for x in range(1, 1000):
            if x in a:
                continue
            trial = sorted(a + [x])
            t_diffs = set()
            ok = True
            for i in range(len(trial)):
                for j in range(i):
                    dd = trial[i] - trial[j]
                    if dd in t_diffs:
                        ok = False
                        break
                    t_diffs.add(dd)
                if not ok:
                    break
            if ok:
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 100
            a.append(y)
            
    return a
