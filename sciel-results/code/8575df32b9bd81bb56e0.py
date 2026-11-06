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

def _search_fast(M, target_diffs, deadline):
    """
    Fast DFS with set-based diff tracking and aggressive pruning.
    """
    target_set = set(target_diffs)
    
    def dfs(current_set, used_diffs, remaining, depth):
        if not remaining:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
        
        last = current_set[-1]
        
        # Generate candidates from remaining diffs
        potential_xs = set()
        for a in current_set:
            for d in remaining:
                x = a + d
                if x > last and x <= M:
                    potential_xs.add(x)
        
        if not potential_xs:
            return None

        # Score candidates: prefer those covering more remaining diffs, then smaller x
        candidates = []
        for x in potential_xs:
            conflict = False
            new_covered = 0
            for a in current_set:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                if d in remaining:
                    new_covered += 1
            
            if conflict:
                continue
            
            candidates.append((x, -new_covered))
        
        if not candidates:
            return None
            
        candidates.sort()
        
        for x, _ in candidates:
            new_diffs = set()
            for a in current_set:
                d = x - a
                new_diffs.add(d)
            
            new_used = used_diffs | new_diffs
            new_remaining = remaining - new_diffs
            
            res = dfs(current_set + [x], new_used, new_remaining, depth + 1)
            if res is not None:
                return res
                
        return None

    return dfs([0], set(), set(target_diffs), 0)

def _greedy_construct():
    """
    Greedy construction: at each step, pick the smallest x that covers the most
    remaining differences without creating conflicts.
    """
    result = [0]
    used_diffs = set()
    remaining = set(range(1, 31))
    
    while remaining and len(result) < 32:
        best_x = None
        best_count = -1
        
        # Try candidate x values from last+1 upward
        last = result[-1]
        # Limit search range for speed
        for x in range(last + 1, last + 200):
            valid = True
            new_covered = 0
            for a in result:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                if d in remaining:
                    new_covered += 1
            
            if valid and new_covered > best_count:
                best_count = new_covered
                best_x = x
                # Early exit if we cover all remaining
                if new_covered == len(remaining):
                    break
        
        if best_x is None or best_count == 0:
            # Try larger range
            for x in range(last + 200, last + 1000):
                valid = True
                new_covered = 0
                for a in result:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    if d in remaining:
                        new_covered += 1
                
                if valid and new_covered > best_count:
                    best_count = new_covered
                    best_x = x
                    if new_covered == len(remaining):
                        break
        
        if best_x is None:
            break
            
        result.append(best_x)
        for a in result[:-1]:
            used_diffs.add(best_x - a)
        remaining -= used_diffs
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # Try progressively larger upper bounds with fast search
    for M in range(35, 150):
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
            
        # Allocate time based on M
        if M < 45:
            m_deadline = min(deadline, time.time() + 10.0)
        elif M < 60:
            m_deadline = min(deadline, time.time() + 6.0)
        elif M < 80:
            m_deadline = min(deadline, time.time() + 4.0)
        else:
            m_deadline = min(deadline, time.time() + 2.0)
        
        result = _search_fast(M, target_diffs, m_deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                return result
                
    # Fallback: greedy construction
    result = _greedy_construct()
    if _check_sidon_and_cover(result, target_diffs):
        return result
    
    # Last resort: simple incremental
    result = [0]
    used_diffs = set()
    remaining = set(range(1, 31))
    x = 1
    while remaining and len(result) < 32 and x < 10000:
        valid = True
        new_diffs = set()
        for a in result:
            d = x - a
            if d in used_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            result.append(x)
            used_diffs.update(new_diffs)
            remaining -= new_diffs
        x += 1
    
    return result
# EVOLVE-BLOCK-END
