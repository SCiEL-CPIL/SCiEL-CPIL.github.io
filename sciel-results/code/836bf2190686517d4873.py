import time
import random
import itertools
from bisect import bisect_left, bisect_right

N = 128
MAX_TIME = 55.0  # safety margin for 60s limit

def _count_conflicts(s_set):
    """
    Returns a dictionary mapping each element in s_set to the number of 
    forbidden triples (a,b,c) in s_set that contain that element.
    A forbidden triple is a,b,c distinct in s_set such that a+b, a+c, b+c are in s_set.
    """
    s_list = sorted(s_set)
    n = len(s_list)
    counts = {x: 0 for x in s_list}
    
    # Iterate over all triples a < b < c
    # Optimization: if a+b > 128, then a+c > 128, so no sums can be in s_set (since max is 128).
    # Actually, sums just need to be in s_set, which is subset of 1..128.
    # So if a+b > 128, it's impossible for a+b to be in s_set.
    # Thus, we only need to check triples where a+b <= 128.
    
    for i in range(n):
        a = s_list[i]
        # If a + s_list[i+1] > 128, no further b will work for this a
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
            
        for j in range(i + 1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if ab not in s_set:
                continue
            # Now check c > b
            # We need a+c in s_set and b+c in s_set
            # c must be such that a+c <= 128 and b+c <= 128
            # So c <= 128 - b
            # Also c > b
            # Iterate c from b+1 to min(128-b, max_val_in_s)
            # We can just iterate k from j+1 to n
            for k in range(j + 1, n):
                c = s_list[k]
                if b + c > 128:
                    break
                # Check if a+c and b+c are in s_set
                # We already know ab is in s_set
                if a + c in s_set and b + c in s_set:
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def _greedy_remove_until_valid(s_set, start_time):
    """
    Start with s_set and remove elements with highest conflict count until valid.
    """
    s_set = set(s_set)
    while True:
        if time.time() - start_time > MAX_TIME - 5.0:
            break
        counts = _count_conflicts(s_set)
        if not counts or all(v == 0 for v in counts.values()):
            break # Valid
        
        # Find element with max count
        max_count = max(counts.values())
        candidates = [x for x, v in counts.items() if v == max_count]
        # Tie-break: remove smallest to keep the set "dense" at the top? 
        # Or remove the largest? 
        # Let's try removing the one with max count. If tie, pick random or smallest.
        # Let's try smallest to preserve the "top half" structure which is known to be good.
        remove_el = min(candidates)
        s_set.remove(remove_el)
        
    return s_set

def _fast_validity_check(s):
    """Faster check using precomputed pair sums."""
    s_set = set(s)
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
            if ab not in s_set:
                continue
            for k in range(j+1, n):
                c = s_list[k]
                if b + c > 128: break
                if a + c in s_set and b + c in s_set:
                    return False
    return True

def _try_greedy_add(s_set, start_time):
    """Try to greedily add elements to improve the set."""
    s_set = set(s_set)
    candidates = [x for x in range(1, N+1) if x not in s_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - 3.0:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            new_s = s_set | {cand}
            if _fast_validity_check(new_s):
                s_set = new_s
                candidates = [x for x in candidates if x != cand]
                improved = True
    
    return sorted(s_set)

def _try_remove_add(s_set, start_time):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        # Start with remaining_set and greedily add
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - 3.0:
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
    # This was measured to yield 1.25 in knowledge.
    
    # 1. Greedy Removal Phase
    full_set = set(range(1, N+1))
    s_set = _greedy_remove_until_valid(full_set, start_time)
    
    # 2. Greedy Add Phase
    if time.time() - start_time < MAX_TIME - 5.0:
        s_set = set(_try_greedy_add(s_set, start_time))
        
    # 3. Local Search (Remove-Add) Phase
    if time.time() - start_time < MAX_TIME - 3.0:
        s_set = set(_try_remove_add(s_set, start_time))
        
    # 4. Random Perturbation / Restart if time permits
    # We can try a few random restarts of the removal process if time allows
    best_s = sorted(s_set)
    best_len = len(best_s)
    
    if time.time() - start_time < MAX_TIME - 10.0:
        for _ in range(10): # Max 10 restarts
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            # Randomize tie-breaking in removal? 
            # Or just run the same deterministic process? 
            # The deterministic process is stable. 
            # Let's try perturbing the initial set or the removal order.
            # Since the removal is deterministic based on counts, it will find the same local optimum.
            # To explore, we can randomly remove one extra element before starting greedy removal.
            perturbed_set = set(full_set)
            if perturbed_set:
                random_remove = random.choice(list(perturbed_set))
                perturbed_set.remove(random_remove)
            
            s_set = _greedy_remove_until_valid(perturbed_set, start_time)
            
            if time.time() - start_time < MAX_TIME - 3.0:
                s_set = set(_try_greedy_add(s_set, start_time))
                
            if time.time() - start_time < MAX_TIME - 1.0:
                s_set = set(_try_remove_add(s_set, start_time))
                
            if len(s_set) > best_len:
                best_s = sorted(s_set)
                best_len = len(best_s)
                
    # Final validation
    if not _fast_validity_check(set(best_s)):
        # Fallback to known valid set (top half)
        best_s = list(range(63, 129))
        
    return [int(x) for x in best_s]
