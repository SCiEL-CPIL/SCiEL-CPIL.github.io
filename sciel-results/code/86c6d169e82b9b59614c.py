"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time
from itertools import combinations

_START = time.time()
DEADLINE = 82  # leave 8s margin before 90s kill

def construct():
    t0 = time.time()
    
    # Generate all smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):
        v2 = 1 << a
        for b in range(0, 6):
            v23 = v2 * (3 ** b)
            if v23 > 128:
                break
            for c in range(0, 5):
                v = v23 * (5 ** c)
                if v > 128:
                    break
                if v >= 1:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    smooth_idx = {s: i for i, s in enumerate(smooth)}
    
    # Precompute smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True
    
    # Precompute pair antichain validity
    pair_ok = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                pair_ok[i][j] = False
                pair_ok[j][i] = False
    
    # Precompute compatible masks
    compatible = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(i + 1, n_smooth):
            if pair_ok[i][j]:
                mask |= (1 << j)
        compatible[i] = mask
    
    # Precompute minimal term count for each target using increasing k
    # best_terms[target] = list of smooth values (or None)
    best_terms = [None] * 129
    best_count = [999] * 129
    
    # For each target, find minimal representation
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            break
        
        # Try 1-term
        if is_sm[n]:
            best_terms[n] = [n]
            best_count[n] = 1
            continue
        
        # Try 2-term
        found = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 >= n:
                break
            s2 = n - s1
            if s2 <= s1:
                continue
            if not is_sm[s2]:
                continue
            j = smooth_idx.get(s2, -1)
            if j >= 0 and pair_ok[i][j]:
                best_terms[n] = [s1, s2]
                best_count[n] = 2
                found = True
                break
        if found:
            continue
        
        # Try 3-term
        found = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 * 3 >= n:
                break
            rem1 = n - s1
            for j in range(i + 1, n_smooth):
                s2 = smooth[j]
                if s2 * 2 >= rem1:
                    break
                s3 = rem1 - s2
                if s3 <= s2:
                    break
                if not is_sm[s3]:
                    continue
                k = smooth_idx.get(s3, -1)
                if k >= 0 and pair_ok[i][j] and pair_ok[i][k] and pair_ok[j][k]:
                    best_terms[n] = [s1, s2, s3]
                    best_count[n] = 3
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # Try 4-term
        found = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 * 4 >= n:
                break
            rem1 = n - s1
            for j in range(i + 1, n_smooth):
                s2 = smooth[j]
                if s2 * 3 >= rem1:
                    break
                rem2 = rem1 - s2
                for k in range(j + 1, n_smooth):
                    s3 = smooth[k]
                    if s3 * 2 >= rem2:
                        break
                    s4 = rem2 - s3
                    if s4 <= s3:
                        break
                    if not is_sm[s4]:
                        continue
                    l = smooth_idx.get(s4, -1)
                    if (l >= 0 and pair_ok[i][j] and pair_ok[i][k] and 
                        pair_ok[i][l] and pair_ok[j][k] and 
                        pair_ok[j][l] and pair_ok[k][l]):
                        best_terms[n] = [s1, s2, s3, s4]
                        best_count[n] = 4
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue
        
        # Try 5-term with backtracking
        found = False
        search_cache = {}
        
        def _search5(rem, min_idx, chosen, depth, max_depth):
            if depth == max_depth:
                return list(chosen) if rem == 0 else None
            if rem <= 0:
                return None
            
            # Pruning: min possible sum with remaining terms
            min_possible = 0
            cnt = 0
            for idx in range(min_idx, n_smooth):
                if cnt >= max_depth - depth:
                    break
                min_possible += smooth[idx]
                cnt += 1
            if min_possible > rem:
                return None
            
            # Pruning: max possible sum
            max_possible = 0
            cnt = 0
            for idx in range(n_smooth - 1, min_idx - 1, -1):
                if idx < min_idx:
                    break
                if cnt >= max_depth - depth:
                    break
                max_possible += smooth[idx]
                cnt += 1
            if max_possible < rem:
                return None
            
            cache_key = (rem, min_idx, tuple(chosen), depth)
            if cache_key in search_cache:
                return search_cache[cache_key]
            
            result = None
            for idx in range(min_idx, n_smooth):
                s = smooth[idx]
                if s > rem:
                    break
                
                # Check antichain with all chosen
                ok = True
                for c in chosen:
                    if s % c == 0 or c % s == 0:
                        ok = False
                        break
                if not ok:
                    continue
                
                chosen.append(s)
                sub = _search5(rem - s, idx + 1, chosen, depth + 1, max_depth)
                if sub is not None:
                    result = sub
                    break
                chosen.pop()
            
            search_cache[cache_key] = result
            return result
        
        for max_terms in range(5, 11):
            if time.time() - t0 > DEADLINE:
                break
            search_cache.clear()
            result = _search5(n, 0, [], 0, max_terms)
            if result is not None:
                best_terms[n] = result
                best_count[n] = max_terms
                found = True
                break
        
        if not found:
            # Try 6+ terms
            for max_terms in range(11, 31):
                if time.time() - t0 > DEADLINE:
                    break
                search_cache.clear()
                result = _search5(n, 0, [], 0, max_terms)
                if result is not None:
                    best_terms[n] = result
                    best_count[n] = max_terms
                    found = True
                    break
        
        if not found:
            # Last resort fallback
            if is_sm[n]:
                best_terms[n] = [n]
                best_count[n] = 1
            else:
                # Try any valid pair
                for i in range(n_smooth):
                    s1 = smooth[i]
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if not is_sm[s2]:
                        continue
                    j = smooth_idx.get(s2, -1)
                    if j >= 0 and pair_ok[i][j]:
                        best_terms[n] = [s1, s2]
                        best_count[n] = 2
                        found = True
                        break
                if not found:
                    best_terms[n] = [1, n - 1] if n > 1 else [1]
                    best_count[n] = 2 if n > 1 else 1
    
    # Fill any missing targets
    for n in range(1, 129):
        if best_terms[n] is None:
            if is_sm[n]:
                best_terms[n] = [n]
                best_count[n] = 1
            else:
                for i in range(n_smooth):
                    s1 = smooth[i]
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if not is_sm[s2]:
                        continue
                    j = smooth_idx.get(s2, -1)
                    if j >= 0 and pair_ok[i][j]:
                        best_terms[n] = [s1, s2]
                        best_count[n] = 2
                        break
                if best_terms[n] is None:
                    best_terms[n] = [1, n - 1] if n > 1 else [1]
                    best_count[n] = 2 if n > 1 else 1
    
    return best_terms[1:129]
