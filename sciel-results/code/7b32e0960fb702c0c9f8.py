"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_k4s():
    """Precompute all C(16,4) = 1820 sets of 4 vertices, each as a tuple of 6 edge indices."""
    n = 16
    # Map (i,j) to index in the 120-element list
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            idx += 1
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(tuple(edge_idx[e] for e in edges))
    return k4s, edge_idx

_K4S, _EDGE_IDX = _precompute_k4s()
_N_K4 = len(_K4S)  # 1820


def _count_colors(coloring):
    """Return the number of distinct colors used."""
    return len(set(coloring))


def _check_validity(coloring):
    """Check if every K4 has at least 5 distinct colors. Returns (valid, min_colors_in_k4)."""
    min_colors = 10
    for eidx in _K4S:
        colors = {coloring[eidx[k]] for k in range(6)}
        cnt = len(colors)
        if cnt < min_colors:
            min_colors = cnt
        if cnt < 5:
            return False, cnt
    return True, min_colors


def _fast_check_violations(coloring):
    """Return list of K4 indices that violate the constraint (fewer than 5 colors)."""
    violations = []
    for ki, eidx in enumerate(_K4S):
        c0 = coloring[eidx[0]]
        c1 = coloring[eidx[1]]
        c2 = coloring[eidx[2]]
        c3 = coloring[eidx[3]]
        c4 = coloring[eidx[4]]
        c5 = coloring[eidx[5]]
        # Count distinct among 6
        if c0 == c1 == c2 == c3 == c4 == c5:
            violations.append(ki)
        elif c0 == c1 == c2 == c3 == c4:
            # 5 same, 1 different -> 2 colors
            violations.append(ki)
        elif c0 == c1 == c2 == c3:
            # Check if c4,c5 add at most 1 new color
            if c4 == c5:
                violations.append(ki)
            else:
                # 4 same + 2 different = 3 colors
                violations.append(ki)
        else:
            # More complex; just count
            distinct = len({c0, c1, c2, c3, c4, c5})
            if distinct < 5:
                violations.append(ki)
    return violations


def _objective(coloring, palette_size):
    """
    Objective: minimize palette_size + penalty for violations.
    We want palette_size small and zero violations.
    """
    violations = _fast_check_violations(coloring)
    return palette_size + 10 * len(violations)


def _initial_coloring(palette_size):
    """Random coloring with given palette size."""
    return [random.randint(0, palette_size - 1) for _ in range(120)]


def _local_search(coloring, palette_size, max_iter=5000, deadline=None):
    """
    Local search: try to reduce violations, then reduce palette.
    Returns (coloring, palette_size).
    """
    best_coloring = list(coloring)
    best_obj = _objective(best_coloring, palette_size)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random edge to recolor
        e = random.randint(0, 119)
        old_c = best_coloring[e]
        new_c = random.randint(0, palette_size - 1)
        if new_c == old_c:
            continue
        best_coloring[e] = new_c
        
        new_obj = _objective(best_coloring, palette_size)
        
        if new_obj <= best_obj:
            best_obj = new_obj
        else:
            best_coloring[e] = old_c
    
    return best_coloring, palette_size


def _try_reduce_palette(coloring, palette_size, max_iter=3000, deadline=None):
    """
    Try to reduce palette_size by 1: recolor edges to eliminate use of color (palette_size-1).
    """
    if palette_size <= 1:
        return coloring, palette_size
    
    target_color = palette_size - 1
    new_palette = palette_size - 1
    coloring = list(coloring)
    
    # Find edges using target_color
    edges_to_recolor = [i for i, c in enumerate(coloring) if c == target_color]
    
    if not edges_to_recolor:
        return coloring, new_palette
    
    for attempt in range(max_iter):
        if deadline and time.time() > deadline:
            break
        if not edges_to_recolor:
            break
        
        e = random.choice(edges_to_recolor)
        old_c = coloring[e]
        new_c = random.randint(0, new_palette - 1)
        coloring[e] = new_c
        
        # Check validity
        valid, _ = _check_validity(coloring)
        if valid:
            # Check if target_color is still used
            if target_color not in coloring:
                return coloring, new_palette
            edges_to_recolor = [i for i, c in enumerate(coloring) if c == target_color]
        else:
            coloring[e] = old_c
    
    return coloring, palette_size


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Start with the known 17-color solution
    best_coloring = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_palette = 17
    
    # Verify initial
    valid, _ = _check_validity(best_coloring)
    if not valid:
        # Should not happen, but fallback
        best_palette = _count_colors(best_coloring)
    
    # Try to reduce palette from 17 down to 13
    for target_palette in range(16, 12, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        max_restarts = 50 if target_palette >= 14 else 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random coloring of target_palette colors
            coloring = _initial_coloring(target_palette)
            
            # Local search to fix violations
            coloring, _ = _local_search(coloring, target_palette, max_iter=3000, deadline=deadline)
            
            # Check if valid
            valid, _ = _check_validity(coloring)
            if valid:
                found = True
                best_coloring = coloring
                best_palette = target_palette
                break
            
            # Also try starting from the current best and reducing
            if restart == 0:
                coloring = list(best_coloring)
                # Recolor to fit in target_palette
                for i in range(120):
                    if coloring[i] >= target_palette:
                        coloring[i] = random.randint(0, target_palette - 1)
                coloring, _ = _local_search(coloring, target_palette, max_iter=5000, deadline=deadline)
                valid, _ = _check_validity(coloring)
                if valid:
                    found = True
                    best_coloring = coloring
                    best_palette = target_palette
                    break
        
        if found:
            continue
        else:
            # If we couldn't achieve target_palette, stop trying lower
            break
    
    # Final check
    valid, _ = _check_validity(best_coloring)
    if not valid:
        # Fallback to the original 17-color solution
        best_coloring = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        best_palette = 17
    
    return best_coloring

# EVOLVE-BLOCK-END
