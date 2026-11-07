import time
import random

N = 128
MAX_TIME = 85.0  # safety margin for 90s wall-clock limit

def _make_bitset(s_set):
    """Build a bitset where bit (x-1) is set if x is in s_set."""
    bitset = 0
    for x in s_set:
        bitset |= (1 << (x - 1))
    return bitset

def _in_bitset(bitset, val):
    """Check if val is in the set represented by bitset."""
    if val < 1 or val > 128:
        return False
    return (bitset >> (val - 1)) & 1

def _count_conflicts(s_set):
    """
    Returns a dictionary mapping each element in s_set to the number of 
    forbidden triples (a,b,c) in s_set that contain that element.
    A forbidden triple is a,b,c distinct in s_set such that a+b, a+c, b+c are in s_set.
    """
    bitset = _make_bitset(s_set)
    s_list = sorted(s_set)
    n = len(s_list)
    counts = {x: 0 for x in s_list}
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
            
        for j in range(i + 1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if not _in_bitset(bitset, ab):
                continue
            for k in range(j + 1, n):
                c = s_list[k]
                bc = b + c
                if bc > 128:
                    break
                ac = a + c
                if _in_bitset(bitset, ac) and _in_bitset(bitset, bc):
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def _fast_validity_check(s):
    """Faster check using precomputed bitset for O(1) membership."""
    s_set = set(s)
    bitset = _make_bitset(s_set)
    s_list = sorted(s)
    n = len(s_list)
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n: break
        if a + s_list[i+1] > 128: break
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128: break
            if not _in_bitset(bitset, ab):
                continue
            for k in range(j+1, n):
                c = s_list[k]
                if b + c > 128: break
                if _in_bitset(bitset, a + c) and _in_bitset(bitset, b + c):
                    return False
    return True

def _greedy_remove_until_valid(s_set, start_time, deadline_margin=5.0):
    """
    Start with s_set and remove elements with highest conflict count until valid.
    """
    s_set = set(s_set)
    while True:
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        counts = _count_conflicts(s_set)
        if not counts or all(v == 0 for v in counts.values()):
            break  # Valid
        
        # Find element with max count
        max_count = max(counts.values())
        candidates = [x for x, v in counts.items() if v == max_count]
        # Tie-break: remove smallest to preserve top-half density
        remove_el = min(candidates)
        s_set.remove(remove_el)
        
    return s_set

def _try_greedy_add(s_set, start_time, deadline_margin=5.0):
    """Try to greedily add elements to improve the set."""
    s_set = set(s_set)
    candidates = [x for x in range(1, N+1) if x not in s_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - deadline_margin:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            new_s = s_set | {cand}
            if _fast_validity_check(new_s):
                s_set = new_s
                candidates = [x for x in candidates if x != cand]
                improved = True
    
    return sorted(s_set)

def _try_remove_add(s_set, start_time, deadline_margin=5.0):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            new_s = current | {cand}
            if _fast_validity_check(new_s):
                current = new_s
        
        if len(current) > best_len:
            best_s = current
            best_len = len(current)
    
    return sorted(best_s)

def construct():
    start_time = time.time()
    
    # Strategy: Start with full set, greedily remove conflicts, then add back.
    
    # 1. Greedy Removal Phase
    full_set = set(range(1, N+1))
    s_set = _greedy_remove_until_valid(full_set, start_time)
    
    # 2. Greedy Add Phase
    if time.time() - start_time < MAX_TIME - 5.0:
        s_set = set(_try_greedy_add(s_set, start_time))
        
    # 3. Local Search (Remove-Add) Phase
    if time.time() - start_time < MAX_TIME - 5.0:
        s_set = set(_try_remove_add(s_set, start_time))
        
    best_s = sorted(s_set)
    best_len = len(best_s)
    
    # 4. Random Perturbation / Restart if time permits
    if time.time() - start_time < MAX_TIME - 10.0:
        for _ in range(20):  # More restarts possible with faster bitset ops
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            # Randomly remove 1-3 elements before starting greedy removal
            perturbed_set = set(full_set)
            for _ in range(random.randint(1, 3)):
                if perturbed_set:
                    random_remove = random.choice(list(perturbed_set))
                    perturbed_set.remove(random_remove)
            
            s_set = _greedy_remove_until_valid(perturbed_set, start_time)
            
            if time.time() - start_time < MAX_TIME - 5.0:
                s_set = set(_try_greedy_add(s_set, start_time))
                
            if time.time() - start_time < MAX_TIME - 2.0:
                s_set = set(_try_remove_add(s_set, start_time))
                
            if len(s_set) > best_len:
                best_s = sorted(s_set)
                best_len = len(s_set)
                
    # Final validation
    if not _fast_validity_check(set(best_s)):
        # Fallback to known valid set (top half)
        best_s = list(range(63, 129))
        
    return [int(x) for x in best_s]
