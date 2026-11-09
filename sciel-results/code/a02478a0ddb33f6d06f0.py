"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _precompute_indices(n):
    """Precompute for red K4 and blue K5 checks."""
    # For each 4-subset, list the 4 triple indices
    red_4sets = []
    red_trip_indices = []
    for quad in combinations(range(n), 4):
        tris = list(combinations(quad, 3))
        idx = []
        for t in tris:
            # Find index of (a,b,c) in lexicographic triple list
            # We'll use a lookup dict
            idx.append((t[0], t[1], t[2]))
        red_trip_indices.append(idx)
    # For each 5-subset, list the 10 triple indices
    blue_5sets = []
    blue_trip_indices = []
    for quint in combinations(range(n), 5):
        tris = list(combinations(quint, 3))
        idx = []
        for t in tris:
            idx.append((t[0], t[1], t[2]))
        blue_trip_indices.append(idx)
    return red_trip_indices, blue_trip_indices

def _triple_index(a, b, c, n):
    """Compute lexicographic index of triple (a,b,c) where a<b<c."""
    # Count triples (x,y,z) with x<b, plus triples with x==b, y<c, plus (a,b,c) offset
    # Standard formula: index = C(b,3) + C(c-b-1,2) + ... 
    # Actually for lexicographic order of (a,b,c) with a<b<c:
    # index = sum_{a' < a} C(n-1-a', 2) + sum_{b' : a<b'<b} C(n-1-b', 1) + c - b - 1
    # But this is complex. Let's just use a precomputed lookup.
    
def _count_violations(colors, n, red_trip_indices, blue_trip_indices, triple_lookup):
    """Count red K4 violations and blue K5 violations."""
    red_viol = 0
    for idx_list in red_trip_indices:
        all_red = True
        for t in idx_list:
            if colors[triple_lookup[t]] != 0:
                all_red = False
                break
        if all_red:
            red_viol += 1
    blue_viol = 0
    for idx_list in blue_trip_indices:
        all_blue = True
        for t in idx_list:
            if colors[triple_lookup[t]] != 1:
                all_blue = False
                break
        if all_blue:
            blue_viol += 1
    return red_viol, blue_viol

def _random_coloring(n, seed=None):
    """Generate a random balanced coloring."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    m = n * (n-1) * (n-2) // 6
    colors = rng.randint(0, 2, size=m).tolist()
    return colors

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start with known small valid coloring, try to grow
    # For n=9, use the mod-3 construction
    n_start = 9
    m_start = n_start * (n_start-1) * (n_start-2) // 6
    colors_start = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_start), 3)]
    
    best_n = n_start
    best_colors = colors_start
    
    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        m = target_n * (target_n-1) * (target_n-2) // 6
        # Precompute triple lookup
        triple_list = list(combinations(range(target_n), 3))
        triple_lookup = {t: i for i, t in enumerate(triple_list)}
        red_trip_indices, blue_trip_indices = _precompute_indices(target_n)
        
        # Start from the previous best coloring, extend with random
        if len(best_colors) < m:
            colors = best_colors + [0] * (m - len(best_colors))
        else:
            colors = best_colors[:m]
        
        # Local search to eliminate violations
        max_iter = 50000
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            red_viol, blue_viol = _count_violations(colors, target_n, red_trip_indices, blue_trip_indices, triple_lookup)
            
            if red_viol == 0 and blue_viol == 0:
                best_n = target_n
                best_colors = colors
                break
            
            # Pick a violating structure and flip a triple in it
            if red_viol > 0:
                # Pick first red K4
                for idx_list in red_trip_indices:
                    all_red = all(colors[triple_lookup[t]] == 0 for t in idx_list)
                    if all_red:
                        # Flip one of the triples
                        t = idx_list[np.random.randint(4)]
                        colors[triple_lookup[t]] ^= 1
                        break
            else:
                # Pick first blue K5
                for idx_list in blue_trip_indices:
                    all_blue = all(colors[triple_lookup[t]] == 1 for t in idx_list)
                    if all_blue:
                        t = idx_list[np.random.randint(10)]
                        colors[triple_lookup[t]] ^= 1
                        break
        
        if best_n >= target_n:
            continue
        
        # If we couldn't fix it, try a fresh random start
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            colors = _random_coloring(target_n)
            max_iter = 20000
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                red_viol, blue_viol = _count_violations(colors, target_n, red_trip_indices, blue_trip_indices, triple_lookup)
                
                if red_viol == 0 and blue_viol == 0:
                    best_n = target_n
                    best_colors = colors
                    break
                
                if red_viol > 0:
                    for idx_list in red_trip_indices:
                        all_red = all(colors[triple_lookup[t]] == 0 for t in idx_list)
                        if all_red:
                            t = idx_list[np.random.randint(4)]
                            colors[triple_lookup[t]] ^= 1
                            break
                else:
                    for idx_list in blue_trip_indices:
                        all_blue = all(colors[triple_lookup[t]] == 1 for t in idx_list)
                        if all_blue:
                            t = idx_list[np.random.randint(10)]
                            colors[triple_lookup[t]] ^= 1
                            break
            
            if best_n >= target_n:
                break
        
        if time.time() > deadline - 5:
            break
    
    # Fallback
    if best_n < 3:
        best_n = 3
        best_colors = [0, 0, 0]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
