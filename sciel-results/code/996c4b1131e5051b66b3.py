"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _check_sidon_and_cover(a, target_diffs):
    """Check if a is a Sidon set covering target_diffs."""
    n = len(a)
    if n > 32:
        return False
    if max(a) > 10**6:
        return False
    if 0 not in a:
        return False
    
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d <= 0:
                return False
            if d in diffs:
                return False
            diffs.add(d)
    return all(d in diffs for d in target_diffs)

def _search_bipartite(M, target_diffs, deadline):
    """
    Branch-and-bound on bipartite graph: left=differences, right=candidates.
    Assign elements in increasing order; use bitmask for used diffs.
    Strong pruning: check if remaining diffs can be covered by remaining slots.
    """
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
    
    # Precompute: for each candidate x in 1..M, which diffs it would create
    # with respect to {0} initially, but we need dynamic computation
    # Instead, we'll compute on the fly with bitmasks
    
    best_solution = [None]
    
    def dfs(current_set, used_diffs_mask, remaining_mask, depth):
        if remaining_mask == 0:
            sol = current_set[:]
            if best_solution[0] is None or max(sol) < max(best_solution[0]):
                best_solution[0] = sol
            return True
        
        if len(current_set) >= 32:
            return False
            
        if time.time() > deadline:
            return False
        
        last = current_set[-1]
        n_remaining_slots = 32 - len(current_set)
        
        # Pruning: each new element can cover at most (len(current_set)+1) new diffs
        # (differences to all existing elements + 0, but 0 is already in set)
        # Actually each new element x creates diffs x-a for a in current_set
        # So at most len(current_set) new diffs per element
        # Total coverable = sum over remaining slots of max possible
        # Conservative: each slot covers at most len(current_set) + slot_index diffs
        # Simpler bound: remaining bits must be <= n_remaining_slots * (len(current_set) + n_remaining_slots)
        remaining_count = bin(remaining_mask).count('1')
        max_coverable = 0
        for i in range(n_remaining_slots):
            max_coverable += len(current_set) + i
        if remaining_count > max_coverable:
            return False
        
        # Generate candidates: x > last, x <= M
        # For each x, check if x-a is not in used_diffs for all a in current_set
        # And count how many remaining diffs it covers
        
        candidates = []
        for x in range(last + 1, M + 1):
            conflict = False
            new_covered = 0
            for a in current_set:
                d = x - a
                if d > 1000000:  # sanity
                    conflict = True
                    break
                if used_diffs_mask & (1 << d):
                    conflict = True
                    break
                if remaining_mask & (1 << d):
                    new_covered |= (1 << d)
            
            if conflict:
                continue
            
            new_count = bin(new_covered).count('1')
            # Heuristic: prefer x that covers more remaining diffs, then smaller x
            candidates.append((-new_count, x, new_covered))
        
        if not candidates:
            return False
        
        candidates.sort()
        
        for _, x, new_covered in candidates:
            new_diffs_mask = 0
            for a in current_set:
                d = x - a
                new_diffs_mask |= (1 << d)
            
            new_used = used_diffs_mask | new_diffs_mask
            new_remaining = remaining_mask & ~new_diffs_mask
            
            current_set.append(x)
            if dfs(current_set, new_used, new_remaining, depth + 1):
                current_set.pop()
                return True
            current_set.pop()
        
        return False
    
    dfs([0], 0, target_mask, 0)
    
    if best_solution[0] is not None:
        return best_solution[0]
    return None

def _construct_fallback():
    """
    Construct a valid Sidon set covering 1..30 using a systematic approach.
    """
    result = [0]
    used_diffs = set()
    
    target = set(range(1, 31))
    remaining = target.copy()
    
    candidates_to_try = [1, 4, 10, 18, 23, 34, 49, 66, 84, 103, 123, 144]
    
    for x in candidates_to_try:
        if not remaining:
            break
        if x <= result[-1]:
            continue
        valid = True
        new_diffs = set()
        for a in result:
            d = x - a
            if d in used_diffs:
                valid = False
                break
            new_diffs.add(d)
        if not valid:
            continue
        result.append(x)
        used_diffs.update(new_diffs)
        remaining -= new_diffs
    
    if remaining:
        next_x = result[-1] + 1
        while remaining and len(result) < 32:
            valid = True
            new_diffs = set()
            for a in result:
                d = next_x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                result.append(next_x)
                used_diffs.update(new_diffs)
                remaining -= new_diffs
            next_x += 1
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # Try progressively larger upper bounds
    for M in range(35, 200):
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 0.5:
            break
            
        # Allocate time: more for smaller M
        if M < 45:
            m_deadline = min(deadline, time.time() + 15.0)
        elif M < 60:
            m_deadline = min(deadline, time.time() + 10.0)
        elif M < 80:
            m_deadline = min(deadline, time.time() + 5.0)
        else:
            m_deadline = min(deadline, time.time() + 3.0)
        
        result = _search_bipartite(M, target_diffs, m_deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                return result
                
    return _construct_fallback()
# EVOLVE-BLOCK-END
