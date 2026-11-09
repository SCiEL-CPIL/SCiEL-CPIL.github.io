"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time

num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# Precompute all isosceles triangles as a set of frozensets for O(1) lookup
# tri_set contains frozenset({i,j,k}) for every isosceles triple
_tri_set = set()
for i in range(num_points):
    xi, yi = coords[i]
    for j in range(i + 1, num_points):
        xj, yj = coords[j]
        dij2 = (xi - xj) ** 2 + (yi - yj) ** 2
        for k in range(j + 1, num_points):
            xk, yk = coords[k]
            dik2 = (xi - xk) ** 2 + (yi - yk) ** 2
            djk2 = (xj - xk) ** 2 + (yj - yk) ** 2
            if dij2 == dik2 or dij2 == djk2 or dik2 == djk2:
                _tri_set.add(frozenset((i, j, k)))

def _can_add(p, s_set):
    """Check if point p can be added to set s_set without creating an isosceles triangle."""
    if len(s_set) < 2:
        return True
    s_list = list(s_set)
    for idx_a in range(len(s_list)):
        a = s_list[idx_a]
        for idx_b in range(idx_a + 1, len(s_list)):
            b = s_list[idx_b]
            if frozenset((p, a, b)) in _tri_set:
                return False
    return True

def _is_maximal(s_set):
    """Check if s_set is maximal (no point can be added)."""
    if len(s_set) < 2:
        return False
    s_list = list(s_set)
    for q in range(num_points):
        if q in s_set:
            continue
        valid = True
        for idx_a in range(len(s_list)):
            a = s_list[idx_a]
            for idx_b in range(idx_a + 1, len(s_list)):
                b = s_list[idx_b]
                if frozenset((q, a, b)) in _tri_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return False
    return True

def construct():
    start_time = time.time()
    time_limit = 290  # 40s margin for safety
    
    best_size = num_points
    best_set = None
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # 1. Randomized greedy construction
        order = list(range(num_points))
        random.shuffle(order)
        
        current_set = set()
        for p in order:
            if _can_add(p, current_set):
                current_set.add(p)
        
        # 2. Repair to maximal: add any addable points
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            for p in range(num_points):
                if p in current_set:
                    continue
                if _can_add(p, current_set):
                    current_set.add(p)
                    changed = True
        
        # 3. Local search: try removing points to get smaller maximal sets
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            s_list = list(current_set)
            random.shuffle(s_list)
            for p in s_list:
                if time.time() - start_time > time_limit:
                    break
                trial = current_set.copy()
                trial.discard(p)
                if _is_maximal(trial):
                    current_set = trial
                    improved = True
                    break
        
        # Update best
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = set()
        for p in range(num_points):
            if _can_add(p, chosen):
                chosen.add(p)
        # Make maximal
        changed = True
        while changed:
            changed = False
            for p in range(num_points):
                if p not in chosen and _can_add(p, chosen):
                    chosen.add(p)
                    changed = True
        best_set = chosen
    
    return [coords[i] for i in sorted(best_set)]

# EVOLVE-BLOCK-END
