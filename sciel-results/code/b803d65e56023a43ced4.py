import time
import random

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 115  # Safety margin

def _hamming(x, y):
    return (x ^ y).bit_count()

def _is_valid_extension(v, path, used):
    """
    Check if v can be added to the end of path.
    path is a list of vertices.
    used is a set of vertices in path.
    """
    if v in used:
        return False
    
    cur = path[-1]
    if _hamming(v, cur) != 1:
        return False
        
    # Check against all non-consecutive vertices
    # The only consecutive vertex is path[-1].
    # We need to check v against path[0:-1]
    for p in path[:-1]:
        if _hamming(v, p) == 1:
            return False
    return True

def _greedy_warnsdorff(start=0):
    """
    Greedy snake using Warnsdorff's rule:
    Always move to the neighbor with the fewest future options.
    """
    path = [start]
    used = {start}
    
    while True:
        cur = path[-1]
        candidates = []
        
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _hamming(v, cur) != 1: # Redundant but safe
                continue
            # Check induced property
            valid = True
            for p in path[:-1]:
                if _hamming(v, p) == 1:
                    valid = False
                    break
            if valid:
                # Calculate score: number of valid options for v
                # We want to pick the one with MINIMUM options (Warnsdorff)
                score = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used or v2 == cur:
                        continue
                    # Check if v2 is valid for v
                    # v2 must not be in used
                    # v2 must not be adjacent to any p in path[:-1] (excluding v? No, v is not in path yet)
                    # Actually, if we add v, the new path is path + [v].
                    # The next vertex w must be adjacent to v and not adjacent to path[:-1].
                    # So we check if v2 is adjacent to v (yes, by construction) 
                    # and not adjacent to any p in path[:-1].
                    ok = True
                    for p in path[:-1]:
                        if _hamming(v2, p) == 1:
                            ok = False
                            break
                    if ok:
                        score += 1
                candidates.append((score, v))
        
        if not candidates:
            break
            
        # Pick candidate with minimum score. Random tie-break.
        min_score = min(c[0] for c in candidates)
        best_candidates = [c[1] for c in candidates if c[0] == min_score]
        next_v = random.choice(best_candidates)
        
        path.append(next_v)
        used.add(next_v)
        
    return path

def _try_extend_ends(path, used, time_budget=0.05):
    """
    Try to extend the path at either end.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        # Try tail
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _hamming(v, cur) != 1:
                continue
            ok = True
            for p in path[:-1]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                improved = True
                break
        
        if improved:
            continue
            
        # Try head
        cur = path[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _hamming(v, cur) != 1:
                continue
            ok = True
            for p in path[1:]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                path.insert(0, v)
                used.add(v)
                improved = True
                break
                
    return path

def construct():
    start_time = time.time()
    best_path = []
    
    # Initial greedy
    p = _greedy_warnsdorff(0)
    best_path = p
    
    # Randomized restarts
    while time.time() - start_time < TIME_LIMIT:
        start_v = random.randint(0, FULL_MASK)
        p = _greedy_warnsdorff(start_v)
        
        # Local search
        used = set(p)
        p = _try_extend_ends(p, used, time_budget=0.05)
        
        if len(p) > len(best_path):
            best_path = p
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 5:
            break

    return best_path
