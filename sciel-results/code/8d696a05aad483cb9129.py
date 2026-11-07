import time
import random

N = 128
MAX_TIME = 55.0  # safety margin for 60s limit (evaluator kills at 90s, but we want margin)

def _fast_validity_check(s):
    """Faster check using precomputed pair sums."""
    s_set = set(s)
    s_list = sorted(s)
    n = len(s_list)
    
    # For each (a,b) pair, if a+b is in s, check if there's c > b with a+c and b+c in s
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab not in s_set:
                continue
            max_c = min(128 - a, 128 - b)
            for c in range(b+1, max_c+1):
                if c not in s_set:
                    continue
                if a + c in s_set and b + c in s_set:
                    return False
    return True

def _can_add(s_set, cand):
    """Check if adding cand to s_set keeps it valid. O(n^2) but with early termination."""
    s_list = sorted(s_set)
    n = len(s_list)
    
    # New violations must involve cand as one of a, b, or c
    # Case 1: cand is c (largest). Need a, b in s_set with a < b < cand,
    #         and a+b in s_set, a+cand in s_set, b+cand in s_set
    for i in range(n):
        a = s_list[i]
        ac = a + cand
        if ac not in s_set and ac != cand:  # ac can't be cand since a>=1
            continue
        # Actually ac > cand since a >= 1, so ac must be in s_set
        if ac not in s_set:
            continue
        for j in range(i+1, n):
            b = s_list[j]
            if b >= cand:
                break
            bc = b + cand
            if bc not in s_set:
                continue
            ab = a + b
            if ab in s_set:
                return False
    
    # Case 2: cand is b (middle). Need a < cand < c, a in s_set, c in s_set,
    #         a+cand in s_set, cand+c in s_set, a+c in s_set
    for a in s_list:
        if a >= cand:
            break
        ac_cand = a + cand
        if ac_cand not in s_set:
            continue
        # Need c > cand, c in s_set, cand+c in s_set, a+c in s_set
        max_c = min(128 - cand, 128 - a)
        for c in range(cand+1, max_c+1):
            if c not in s_set:
                continue
            if cand + c in s_set and a + c in s_set:
                return False
    
    # Case 3: cand is a (smallest). Need cand < b < c, b, c in s_set,
    #         cand+b in s_set, cand+c in s_set, b+c in s_set
    for j in range(n):
        b = s_list[j]
        if b <= cand:
            continue
        bc_cand = cand + b
        if bc_cand not in s_set:
            continue
        for k in range(j+1, n):
            c = s_list[k]
            cc_cand = cand + c
            if cc_cand not in s_set:
                continue
            if b + c in s_set:
                return False
    
    return True

def _greedy_add(s_set, start_time):
    """Try to greedily add elements to improve the set."""
    s_set = set(s_set)
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - 3.0:
        improved = False
        candidates = [x for x in range(1, N+1) if x not in s_set]
        candidates.sort()
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            if _can_add(s_set, cand):
                s_set.add(cand)
                improved = True
    
    return s_set

def _remove_and_add(s_set, start_time):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    original_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        best_new_set = None
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            if cand == remove_el:
                continue
            new_s = remaining_set | {cand}
            if len(new_s) > original_len and _can_add(remaining_set, cand):
                if best_new_set is None or len(new_s) > len(best_new_set):
                    best_new_set = new_s
        
        if best_new_set is not None:
            return best_new_set
    
    return s_set

def _random_perturbation(s_set, start_time, num_restarts=30):
    """Apply random perturbations to try to find better sets."""
    best_s = set(s_set)
    best_len = len(best_s)
    
    for _ in range(num_restarts):
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        
        s_set = set(best_s)
        
        # Randomly remove 1-3 elements
        num_remove = random.randint(1, 3)
        removable = sorted(s_set)
        if len(removable) <= num_remove:
            continue
        removed = random.sample(removable, num_remove)
        s_set -= set(removed)
        
        # Try to greedily add back
        new_s = _greedy_add(s_set, start_time)
        
        if len(new_s) > best_len:
            best_s = new_s
            best_len = len(new_s)
        
        # Try remove-and-add on current best
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        new_s2 = _remove_and_add(set(best_s), start_time)
        if len(new_s2) > best_len:
            best_s = new_s2
            best_len = len(new_s2)
    
    return best_s

def _hitting_set_removal(start_time):
    """
    Start with full set {1..128} and greedily remove elements that are involved
    in the most forbidden triples. This is the strategy that scored 1.25.
    """
    s_set = set(range(1, N+1))
    
    def count_conflicts(x, s):
        """Count how many forbidden triples involve x."""
        count = 0
        s_list = sorted(s)
        n = len(s_list)
        
        # x as c: need a < b < x, a+b in s, a+x in s, b+x in s
        for i in range(n):
            a = s_list[i]
            if a >= x:
                break
            ax = a + x
            if ax not in s:
                continue
            for j in range(i+1, n):
                b = s_list[j]
                if b >= x:
                    break
                bx = b + x
                if bx not in s:
                    continue
                if a + b in s:
                    count += 1
        
        # x as b: need a < x < c, a+x in s, x+c in s, a+c in s
        for a in s_list:
            if a >= x:
                break
            ax = a + x
            if ax not in s:
                continue
            max_c = min(128 - x, 128 - a)
            for c in range(x+1, max_c+1):
                if c not in s:
                    continue
                if x + c in s and a + c in s:
                    count += 1
        
        # x as a: need x < b < c, x+b in s, x+c in s, b+c in s
        for j in range(n):
            b = s_list[j]
            if b <= x:
                continue
            xb = x + b
            if xb not in s:
                continue
            for k in range(j+1, n):
                c = s_list[k]
                xc = x + c
                if xc not in s:
                    continue
                if b + c in s:
                    count += 1
        
        return count
    
    # Greedily remove the element with the most conflicts
    while not _fast_validity_check(s_set) and time.time() - start_time < MAX_TIME - 5.0:
        # Find the element with the most conflicts
        max_conflicts = -1
        worst_el = None
        for x in sorted(s_set):
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            conflicts = count_conflicts(x, s_set)
            if conflicts > max_conflicts:
                max_conflicts = conflicts
                worst_el = x
            if max_conflicts == 0:
                break
        
        if worst_el is None or max_conflicts <= 0:
            break
        
        s_set.discard(worst_el)
    
    return s_set

def construct():
    start_time = time.time()
    
    # Strategy 1: Hitting set removal from full set (scored 1.25 in observations)
    best_s = _hitting_set_removal(start_time)
    best_len = len(best_s)
    
    # Verify validity
    if not _fast_validity_check(best_s):
        # Fallback to known valid set
        best_s = set(range(63, 129))
        best_len = len(best_s)
    
    # Strategy 2: Greedy add from current best
    if time.time() - start_time < MAX_TIME - 5.0:
        s2 = _greedy_add(set(best_s), start_time)
        if len(s2) > best_len:
            best_s = s2
            best_len = len(s2)
    
    # Strategy 3: Remove-and-add
    if time.time() - start_time < MAX_TIME - 5.0:
        s3 = _remove_and_add(set(best_s), start_time)
        if len(s3) > best_len:
            best_s = s3
            best_len = len(s3)
    
    # Strategy 4: Random perturbation
    if time.time() - start_time < MAX_TIME - 3.0:
        s4 = _random_perturbation(set(best_s), start_time, num_restarts=20)
        if len(s4) > best_len:
            best_s = s4
            best_len = len(s4)
    
    # Final validation
    if not _fast_validity_check(best_s):
        # Fallback to known valid set
        best_s = set(range(63, 129))
    
    # Ensure all are Python ints in a list
    return [int(x) for x in sorted(best_s)]
