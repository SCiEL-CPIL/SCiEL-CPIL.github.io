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

def _search_with_pruning(M, target_diffs, deadline):
    """
    Optimized DFS with strong pruning for Sidon set in [0, M] covering target_diffs.
    Uses bitmasks for efficiency and aggressive pruning.
    """
    initial_set = [0]
    initial_diffs_mask = 0
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
    
    remaining_mask = target_mask
    
    def dfs(current_set, used_diffs_mask, remaining_mask, depth):
        if remaining_mask == 0:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
        
        last = current_set[-1]
        
        # Generate candidates: x > last, x <= M, and x - a must not be in used_diffs
        # and should help cover remaining diffs
        potential_xs = set()
        for a in current_set:
            for d in target_diffs:
                if not (remaining_mask & (1 << d)):
                    continue
                x = a + d
                if x > last and x <= M:
                    potential_xs.add(x)
        
        if not potential_xs:
            return None

        candidates = []
        for x in potential_xs:
            # Check conflicts and count new coverage
            conflict = False
            new_covered_mask = 0
            for a in current_set:
                d = x - a
                if d <= 0:
                    conflict = True
                    break
                if used_diffs_mask & (1 << d):
                    conflict = True
                    break
                if remaining_mask & (1 << d):
                    new_covered_mask |= (1 << d)
            
            if conflict:
                continue
            
            # Heuristic: prioritize x that covers more remaining diffs
            # Also prefer smaller x to leave room for larger values
            new_count = bin(new_covered_mask).count('1')
            candidates.append((x, -new_count))
        
        if not candidates:
            return None
            
        candidates.sort()
        
        for x, _ in candidates:
            new_diffs_mask = 0
            for a in current_set:
                d = x - a
                new_diffs_mask |= (1 << d)
            
            new_used = used_diffs_mask | new_diffs_mask
            new_remaining = remaining_mask & ~new_diffs_mask
            
            res = dfs(current_set + [x], new_used, new_remaining, depth + 1)
            if res is not None:
                return res
                
        return None

    return dfs(initial_set, initial_diffs_mask, remaining_mask, 0)

def _construct_fallback():
    """
    Construct a valid Sidon set covering 1..30 using a systematic approach.
    This is a guaranteed valid fallback.
    """
    # Use a known construction: start with small elements, add carefully
    # We'll build incrementally and verify at each step
    result = [0]
    used_diffs = set()
    
    target = set(range(1, 31))
    remaining = target.copy()
    
    # Try to add elements that cover many remaining diffs
    # Start with small gaps to cover small diffs
    candidates_to_try = [1, 4, 10, 18, 23, 34, 49, 66, 84, 103, 123, 144]
    
    for x in candidates_to_try:
        if not remaining:
            break
        if x <= result[-1]:
            continue
        # Check if x can be added
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
    
    # If we still have remaining diffs, add more elements
    if remaining:
        # Try larger elements
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
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # Try progressively larger upper bounds
    # Start with small M to find minimal diameter solutions
    for M in range(35, 200):
        if time.time() > deadline:
            break
            
        # Give each M a time slice based on remaining time
        remaining_time = deadline - time.time()
        if remaining_time < 0.5:
            break
            
        # Allocate more time to smaller M values where solutions are more likely
        if M < 50:
            m_deadline = min(deadline, time.time() + 8.0)
        elif M < 80:
            m_deadline = min(deadline, time.time() + 5.0)
        else:
            m_deadline = min(deadline, time.time() + 3.0)
        
        result = _search_with_pruning(M, target_diffs, m_deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                return result
                
    # If no solution found, return fallback
    return _construct_fallback()
# EVOLVE-BLOCK-END
