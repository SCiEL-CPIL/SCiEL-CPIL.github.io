"""Combined approach: Greedy construction + Minimality Reduction."""
import time
import random
import itertools

N = 16
DEADLINE = 325.0

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_maximal(S, all_points):
    """
    Check if S is a maximal no-isosceles set.
    S must be no-isosceles (assumed true by construction).
    Maximal: for every p not in S, there exist a, b in S such that {p, a, b} is isosceles.
    """
    S_set = set(S)
    S_list = list(S)
    n_S = len(S_list)
    
    # If S is empty, it's maximal only if grid is empty (not the case).
    # If S has < 2 points, it's not maximal (can always add points).
    if n_S < 2:
        return False
        
    # For each point not in S, check if it is "blocked" by S
    for p in all_points:
        if p in S_set:
            continue
            
        # Check if p forms an isosceles triangle with any pair in S
        blocked = False
        for i in range(n_S):
            if blocked:
                break
            a = S_list[i]
            da = _dist2(p, a)
            for j in range(i + 1, n_S):
                b = S_list[j]
                # We need to check if {p, a, b} is isosceles
                # d_pa is known.
                # d_pb
                d_pb = _dist2(p, b)
                # d_ab
                d_ab = _dist2(a, b)
                
                if da == d_pb or da == d_ab or d_pb == d_ab:
                    blocked = True
                    break
                    
        if not blocked:
            return False
            
    return True

def _reduce_to_minimal(S, all_points):
    """
    Iteratively remove points from S if the remaining set is still maximal.
    This finds a minimal (by inclusion) maximal set.
    """
    S_list = list(S)
    changed = True
    while changed:
        changed = False
        # Try removing each point
        for i in range(len(S_list)):
            p = S_list[i]
            S_test = S_list[:i] + S_list[i+1:]
            
            # Quick check: if len < 2, not maximal
            if len(S_test) < 2:
                continue
                
            if _is_maximal(S_test, all_points):
                S_list = S_test
                changed = True
                break # Restart the scan as the list changed
                
    return S_list

def construct():
    start_time = time.time()
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_total = len(all_points)
    
    best_set = None
    best_size = n_total + 1
    
    rng = random.Random(42)
    
    # Strategy:
    # 1. Generate random maximal sets via greedy.
    # 2. Reduce them to minimal maximal sets.
    # 3. Keep the smallest.
    
    for attempt in range(1000):
        if time.time() - start_time > DEADLINE:
            break
            
        # Greedy construction
        order = all_points[:]
        rng.shuffle(order)
        
        chosen = []
        for p in order:
            # Check if p can be added
            valid = True
            for i in range(len(chosen)):
                if not valid:
                    break
                a = chosen[i]
                da = _dist2(p, a)
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    d_pb = _dist2(p, b)
                    d_ab = _dist2(a, b)
                    if da == d_pb or da == d_ab or d_pb == d_ab:
                        valid = False
                        break
            if valid:
                chosen.append(p)
        
        # The set 'chosen' is maximal by construction.
        # Now reduce it to a minimal maximal set.
        reduced = _reduce_to_minimal(chosen, all_points)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        # Early termination
        if best_size <= 5: # Theoretical lower bound is likely higher, but 5 is a good stop
            break
            
    if best_set is None:
        # Fallback: deterministic greedy
        chosen = []
        for p in all_points:
            valid = True
            for i in range(len(chosen)):
                if not valid: break
                a = chosen[i]
                da = _dist2(p, a)
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    d_pb = _dist2(p, b)
                    d_ab = _dist2(a, b)
                    if da == d_pb or da == d_ab or d_pb == d_ab:
                        valid = False
                        break
            if valid:
                chosen.append(p)
        return _reduce_to_minimal(chosen, all_points)
        
    return best_set
