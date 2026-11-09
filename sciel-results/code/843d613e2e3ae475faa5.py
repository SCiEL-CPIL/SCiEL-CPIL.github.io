"""Bitmask-based local search for 5-colourings of K_n without monochromatic C4."""
import time
import random
import numpy as np

# Deadline guard
_START_TIME = time.time()
DEADLINE = 120.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _count_c4_bitmask(neighbors, n):
    """Count C4 violations using bitmask neighbors.
    neighbors: list of 5 lists, each of length n, containing bitmasks.
    """
    violations = 0
    for c in range(5):
        for u in range(n):
            nu = neighbors[c][u]
            for v in range(u + 1, n):
                nv = neighbors[c][v]
                # Check if they share at least 2 common neighbors
                if (nu & nv).bit_count() >= 2:
                    violations += 1
    return violations

def _local_search_bitmask(n, colors, neighbors, max_iters=2000, timeout=5.0):
    """Local search using bitmask representation."""
    start_t = time.time()
    current_violations = _count_c4_bitmask(neighbors, n)
    
    if current_violations == 0:
        return colors, neighbors, current_violations

    best_violations = current_violations
    best_colors = colors[:]
    best_neighbors = [list(nc) for nc in neighbors]
    
    # Precompute edge indices
    edge_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            edge_idx[(a, b)] = idx
            idx += 1
            
    # List of edges for random selection
    edges = list(edge_idx.keys())
    
    for it in range(max_iters):
        if time.time() - start_t > timeout:
            break
        if _time_left() < 5.0:
            break
        
        # Pick a random edge
        u, v = random.choice(edges)
        eidx = edge_idx[(u, v)]
        old_color = colors[eidx]
        
        # Pick a new color
        new_color = (old_color + random.randint(1, 4)) % 5
        
        # Apply change
        neighbors[old_color][u] ^= (1 << v)
        neighbors[old_color][v] ^= (1 << u)
        neighbors[new_color][u] ^= (1 << v)
        neighbors[new_color][v] ^= (1 << u)
        colors[eidx] = new_color
        
        new_violations = _count_c4_bitmask(neighbors, n)
        
        if new_violations <= current_violations:
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                best_neighbors = [list(nc) for nc in neighbors]
                if current_violations == 0:
                    break
        else:
            # Revert
            neighbors[old_color][u] ^= (1 << v)
            neighbors[old_color][v] ^= (1 << u)
            neighbors[new_color][u] ^= (1 << v)
            neighbors[new_color][v] ^= (1 << u)
            colors[eidx] = old_color
            
    return best_colors, best_neighbors, best_violations

def _try_construct_bitmask(n):
    """Try to find a valid 5-coloring of K_n using bitmask local search."""
    best_violations = float('inf')
    best_colors = None
    best_neighbors = None
    
    # Multiple random restarts
    restarts = 0
    while _time_left() > 10.0:
        restarts += 1
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        neighbors = [[0] * n for _ in range(5)]
        
        # Build initial neighbors
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                c = colors[idx]
                neighbors[c][a] |= (1 << b)
                neighbors[c][b] |= (1 << a)
                idx += 1
                
        colors, neighbors, viol = _local_search_bitmask(n, colors, neighbors, max_iters=5000, timeout=2.0)
        
        if viol < best_violations:
            best_violations = viol
            best_colors = colors
            best_neighbors = neighbors
            
        if best_violations == 0:
            break
            
    if best_violations == 0:
        return best_colors
    return None

def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try n=26 first (known to be feasible)
    if _time_left() > 20.0:
        colors26 = _try_construct_bitmask(26)
        if colors26 is not None:
            best_n = 26
            best_colors = colors26
    
    # Try n=27
    if _time_left() > 15.0:
        colors27 = _try_construct_bitmask(27)
        if colors27 is not None:
            best_n = 27
            best_colors = colors27
    
    # Try n=28
    if _time_left() > 10.0:
        colors28 = _try_construct_bitmask(28)
        if colors28 is not None:
            best_n = 28
            best_colors = colors28
            
    # Try n=29
    if _time_left() > 5.0:
        colors29 = _try_construct_bitmask(29)
        if colors29 is not None:
            best_n = 29
            best_colors = colors29

    return {'n': best_n, 'colors': best_colors}
