import time
import random

def _get_subset_sums(elements):
    """
    Compute all non-empty subset sums.
    Returns None if any duplicate sum is found.
    """
    if not elements:
        return set()
    
    sums = {0}
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            # Check for duplicates
            if ns in sums or ns in new_sums:
                return None
            new_sums.add(ns)
        sums.update(new_sums)
    sums.discard(0)
    return sums

def _is_valid_sums(sums):
    """
    Check if no two distinct values in sums divide each other.
    """
    if not sums:
        return True
    
    arr = sorted(sums)
    max_sum = arr[-1]
    
    for a in arr:
        if 2 * a > max_sum:
            break
        for m in range(2 * a, max_sum + 1, a):
            if m in sums:
                return False
    return True

def _is_valid(elements):
    sums = _get_subset_sums(elements)
    if sums is None:
        return False
    return _is_valid_sums(sums)

def _greedy_construct(candidates, time_limit):
    """
    Greedily add candidates to an empty set.
    """
    current = []
    start_time = time.time()
    random.shuffle(candidates)
    
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        
        new_set = current + [cand]
        if _is_valid(new_set):
            current.append(cand)
            
    return current

def _try_add_two(current, all_candidates, time_limit):
    """
    Try to remove one element and add two to increase size by 1.
    Returns the new set if successful, else None.
    """
    if not current:
        return None
        
    start_time = time.time()
    current_set = set(current)
    
    # Try removing each element
    for i in range(len(current)):
        if time.time() - start_time > time_limit:
            return None
            
        removed = current[i]
        remaining = current[:i] + current[i+1:]
        remaining_set = set(remaining)
        
        # Candidates to add
        cands = [c for c in all_candidates if c not in remaining_set]
        random.shuffle(cands)
        
        # Try pairs of candidates
        # Limit pairs to avoid explosion
        limit = min(50, len(cands))
        for j in range(limit):
            c1 = cands[j]
            for k in range(j+1, limit):
                c2 = cands[k]
                
                if time.time() - start_time > time_limit:
                    return None
                
                new_set = remaining + [c1, c2]
                if _is_valid(new_set):
                    return new_set
                    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_candidates = list(range(1, 257))
    best_result = []
    
    # Initial quick greedy
    baseline = _greedy_construct(all_candidates, time_limit=2.0)
    if len(baseline) > len(best_result):
        best_result = baseline
        
    # Randomized search
    while time.time() < deadline - 10.0:
        # Random restart
        candidates = list(all_candidates)
        random.shuffle(candidates)
        
        # Greedy phase
        current = _greedy_construct(candidates, time_limit=1.0)
        
        if len(current) > len(best_result):
            best_result = list(current)
        
        # Local search phase: Try to increase size
        # 1. Try adding any candidate
        current_set = set(current)
        cands_to_add = [c for c in all_candidates if c not in current_set]
        random.shuffle(cands_to_add)
        
        added = False
        for c in cands_to_add:
            if time.time() > deadline - 5.0:
                break
            new_set = current + [c]
            if _is_valid(new_set):
                current.append(c)
                current_set.add(c)
                added = True
                if len(current) > len(best_result):
                    best_result = list(current)
                break # Restart add loop? Or just continue?
        
        if added:
            continue
            
        # 2. If no single addition works, try remove 1 add 2
        if time.time() < deadline - 5.0:
            improved = _try_add_two(current, all_candidates, time_limit=2.0)
            if improved:
                current = improved
                if len(current) > len(best_result):
                    best_result = list(current)
        
    # Final verification
    if _is_valid(best_result):
        return best_result
    else:
        return [1]

# EVOLVE-BLOCK-END
