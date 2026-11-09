import time
import random
from itertools import combinations

def _is_valid_set(selected):
    """Check if the entire set is valid."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        for u in range(m):
            for v in range(u+1, m):
                for w in range(v+1, m):
                    s = others[u] + others[v] + others[w]
                    if s % a == 0:
                        return False
    return True

def _can_add(a, current):
    """Check if adding a to current keeps it valid. Fast path."""
    if a in current:
        return False
    n = len(current)
    # Check: a must not divide sum of any 3 in current
    if n >= 3:
        for triplet in combinations(current, 3):
            if sum(triplet) % a == 0:
                return False
    # Check: each b in current must not divide (a + c + d) for any 2 others c,d in current
    if n >= 2:
        for i in range(n):
            b = current[i]
            others_no_b = [current[j] for j in range(n) if j != i]
            for pair in combinations(others_no_b, 2):
                s = a + pair[0] + pair[1]
                if s % b == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_set = []
    best_size = 0
    
    # Primary strategy: structured greedy from high range
    # Numbers 144-192: for a in this range, only multiples 2a, 3a are relevant
    # (since sum of 3 numbers in [144,192] is in [435, 573])
    
    # Try multiple random orderings with greedy + local improvement
    for attempt in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Use range 144-192 as base
        pool = list(range(144, 193))
        random.shuffle(pool)
        
        current = []
        for a in pool:
            if time.time() > deadline - 2.0:
                break
            if len(current) >= 48:
                break
            if _can_add(a, current):
                current.append(a)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 48:
                break
        
        # Local search: try to swap or add from extended pool
        if len(current) < 48:
            extended = list(range(120, 193))
            extended = [x for x in extended if x not in current]
            random.shuffle(extended)
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for cand in extended:
                    if time.time() > deadline - 2.0:
                        break
                    if len(current) >= 48:
                        break
                    if cand in current:
                        continue
                    if _can_add(cand, current):
                        current.append(cand)
                        improved = True
                        if len(current) >= 48:
                            break
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                if best_size >= 48:
                    break
        
        # Also try removal-based: start with full 144-192 and remove violators
        if attempt % 5 == 0:
            if time.time() > deadline - 2.0:
                break
            
            current = list(range(144, 193))
            # Remove elements causing violations
            changed = True
            while changed and time.time() < deadline - 2.0:
                changed = False
                n = len(current)
                if n < 4:
                    break
                # Find a violation
                found_violation = False
                for i in range(n):
                    a = current[i]
                    others = [current[j] for j in range(n) if j != i]
                    for triplet in combinations(others, 3):
                        if sum(triplet) % a == 0:
                            # Remove a or one of the triplet
                            # Remove the one with highest "degree" (in most violations)
                            # Simple: remove a
                            current.remove(a)
                            found_violation = True
                            changed = True
                            break
                    if found_violation:
                        break
                if not found_violation:
                    break
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                if best_size >= 48:
                    break
    
    # Final verification
    if best_set and not _is_valid_set(best_set):
        # Fallback: use a known small valid set
        best_set = list(range(180, 193))  # 13 elements, likely valid
        if not _is_valid_set(best_set):
            best_set = list(range(185, 193))
    
    if len(best_set) > 48:
        best_set = best_set[:48]
    
    return best_set
