"""Improved program: multi-start greedy with local search for Property P."""
import time
import random
from itertools import combinations

N = 256
DEADLINE = 80.0  # seconds, safety margin from 90s limit

def _valid_subset(s):
    """Check if subset s satisfies: for all a<b<=c in s, a does not divide b+c."""
    s = sorted(s)
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i+1, n):
            b = s[j]
            for k in range(j, n):
                c = s[k]
                if (b + c) % a == 0:
                    return False
    return True

def _fast_is_valid_with(s_sorted, x):
    """Fast check: can x be added to sorted set s_sorted?
    Only checks triples where x participates.
    Returns True if valid, False otherwise.
    """
    n = len(s_sorted)
    
    # Find where x would go in sorted order
    # We need to check all triples (a,b,c) with a<b<=c where at least one of a,b,c is x
    
    # Case 1: x is the smallest (a=x)
    # Need: for all b,c in s_sorted with x < b <= c, x does NOT divide b+c
    # But we also need b,c > x. Since s_sorted is sorted, find elements > x
    # Actually, a < b <= c, so if a=x, then b > x and c >= b > x
    # Check all pairs (b,c) in s_sorted where b > x and c >= b
    i_start = 0
    # Binary search for first element > x
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s_sorted[mid] > x:
            hi = mid
        else:
            lo = mid + 1
    i_start = lo
    for i in range(i_start, n):
        b = s_sorted[i]
        for j in range(i, n):
            c = s_sorted[j]
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is 'b' (a < x <= c)
    # a from s_sorted with a < x, c from s_sorted with c >= x (c can be x itself since b<=c)
    # Wait, b=x, c>=b=x. c can be x (b=c=x) or c>x from s_sorted
    # If c=x: need a < x and a does NOT divide x+x=2x
    # If c>x: c from s_sorted with c > x, need a < x and a does NOT divide x+c
    # First find elements < x
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s_sorted[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    i_end_less = lo  # elements [0, i_end_less) are < x
    
    # Check c=x (b=c=x): for all a in s_sorted with a < x, a must NOT divide 2x
    for i in range(i_end_less):
        a = s_sorted[i]
        if (2 * x) % a == 0:
            return False
    
    # Check c > x: for all a < x in s_sorted and all c > x in s_sorted
    # Find elements > x
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s_sorted[mid] > x:
            hi = mid
        else:
            lo = mid + 1
    i_start_gt = lo
    for i in range(i_end_less):
        a = s_sorted[i]
        for j in range(i_start_gt, n):
            c = s_sorted[j]
            if (x + c) % a == 0:
                return False
    
    # Case 3: x is 'c' (a < b <= x)
    # a,b from s_sorted with a < b <= x
    # Need: a does NOT divide b+x
    # Find elements <= x
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s_sorted[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    i_end_le = lo  # elements [0, i_end_le) are <= x
    
    for i in range(i_end_le):
        a = s_sorted[i]
        for j in range(i+1, i_end_le):
            b = s_sorted[j]
            if (b + x) % a == 0:
                return False
    
    return True

def _conflict(a, b, c):
    """Check if a divides b+c (the violation condition)."""
    return (b + c) % a == 0

def _is_valid_with(s, x):
    """Check if adding x to sorted set s keeps it valid.
    s should be a sorted list (or we sort it first).
    """
    if not isinstance(s, list):
        s = list(s)
    if s and s[0] != sorted(s)[0]:
        s = sorted(s)
    return _fast_is_valid_with(s, x)

def _greedy_extend(base, candidates):
    """Greedily add elements from candidates to base, maintaining validity."""
    s = list(base)
    s_set = set(s)
    for x in candidates:
        if x in s_set:
            continue
        if _is_valid_with(s, x):
            s.append(x)
            s_set.add(x)
    return s

def _local_search(s, time_limit):
    """Try to improve s by removing one or more elements and adding others."""
    s = sorted(s)
    s_set = set(s)
    best = list(s)
    best_len = len(s)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Single removal + single addition
        for i in range(len(s)):
            if time.time() - start > time_limit:
                break
            removed = s[i]
            new_s = [y for y in s if y != removed]
            new_s_set = set(new_s)
            
            # Try adding each candidate
            for x in range(1, N+1):
                if x in new_s_set:
                    continue
                if _fast_is_valid_with(new_s, x):
                    candidate = new_s + [x]
                    if len(candidate) > best_len:
                        best = sorted(candidate)
                        best_len = len(best)
                        s = best
                        s_set = set(s)
                        improved = True
                        break
            if improved:
                break
        
        # If single swap didn't help, try removing two elements and adding two
        if not improved and len(s) >= 4:
            for i in range(len(s)):
                if time.time() - start > time_limit:
                    break
                for j in range(i+1, len(s)):
                    removed1, removed2 = s[i], s[j]
                    new_s = [y for y in s if y != removed1 and y != removed2]
                    new_s_set = set(new_s)
                    
                    # Try adding two candidates
                    found_pair = False
                    for x1 in range(1, N+1):
                        if time.time() - start > time_limit:
                            break
                        if x1 in new_s_set:
                            continue
                        temp_s = new_s + [x1]
                        temp_set = new_s_set | {x1}
                        for x2 in range(x1+1, N+1):
                            if x2 in temp_set:
                                continue
                            if _fast_is_valid_with(temp_s, x2):
                                candidate = temp_s + [x2]
                                if len(candidate) > best_len:
                                    best = sorted(candidate)
                                    best_len = len(best)
                                    s = best
                                    s_set = set(s)
                                    improved = True
                                    found_pair = True
                                    break
                        if found_pair:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Start with the known valid upper tail
    base = list(range(192, 257))
    
    # Verify base is valid
    if not _valid_subset(base):
        # Fallback: just use a small valid set
        return list(range(200, 257))
    
    best = list(base)
    best_len = len(base)
    
    # Try multiple strategies to extend
    random.seed(42)
    
    # Strategy 1: Try adding elements in descending order from 191 down to 1
    s = list(base)
    for x in range(191, 0, -1):
        if _fast_is_valid_with(s, x):
            s.append(x)
    if len(s) > best_len:
        best = s
        best_len = len(s)
    
    # Strategy 2: Try adding elements in ascending order
    s = list(base)
    for x in range(1, 192):
        if _fast_is_valid_with(s, x):
            s.append(x)
    if len(s) > best_len:
        best = s
        best_len = len(s)
    
    # Strategy 3: Random order - more trials
    candidates = list(range(1, 192))
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 15:
            break
        random.shuffle(candidates)
        s = list(base)
        for x in candidates:
            if _fast_is_valid_with(s, x):
                s.append(x)
        if len(s) > best_len:
            best = s
            best_len = len(s)
    
    # Strategy 4: Greedy with priority by "conflict count" 
    # Prefer elements that conflict with fewer existing elements
    s = list(base)
    s_set = set(s)
    remaining = [x for x in range(1, 192) if x not in s_set]
    while remaining and time.time() - start_time < DEADLINE - 15:
        # Score each candidate by how many conflicts it has
        best_x = None
        min_conflicts = float('inf')
        for x in remaining:
            conflicts = 0
            # Count conflicts if we add x
            n = len(s)
            # Case 1: x as a
            for i in range(n):
                b = s[i]
                if b > x:
                    for j in range(i, n):
                        c = s[j]
                        if c >= b and (b + c) % x == 0:
                            conflicts += 1
            # Case 2: x as b (a < x <= c)
            for i in range(n):
                a = s[i]
                if a < x:
                    # c = x
                    if (2*x) % a == 0:
                        conflicts += 1
                    # c > x
                    for j in range(n):
                        c = s[j]
                        if c > x and (x + c) % a == 0:
                            conflicts += 1
            # Case 3: x as c (a < b <= x)
            for i in range(n):
                a = s[i]
                for j in range(i+1, n):
                    b = s[j]
                    if b <= x and (b + x) % a == 0:
                        conflicts += 1
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_x = x
            if conflicts == 0:
                break
        
        if best_x is None or min_conflicts > 0:
            break
        
        s.append(best_x)
        s_set.add(best_x)
        remaining.remove(best_x)
    
    if len(s) > best_len:
        best = s
        best_len = len(s)
    
    # Strategy 5: Local search on the best found so far
    if time.time() - start_time < DEADLINE - 5:
        remaining_time = DEADLINE - (time.time() - start_time) - 3
        best = _local_search(best, min(remaining_time, 30))
        best_len = len(best)
    
    # Final verification
    best = sorted(best)
    if not _valid_subset(best):
        # Fallback to a definitely valid subset
        return list(range(200, 257))
    
    return best

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
