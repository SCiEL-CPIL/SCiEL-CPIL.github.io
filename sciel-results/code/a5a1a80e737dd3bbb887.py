"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge[i, j] = colors[idx]
            edge[j, i] = colors[idx]
            idx += 1
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    cs = np.array([edge[a,b], edge[a,c], edge[a,d],
                                   edge[b,c], edge[b,d], edge[c,d]], dtype=np.int32)
                    if len(np.unique(cs)) < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def _try_multiplicative(p, values):
    """Color edge (i,j) by (values[i] * values[j]) % p."""
    colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            colors.append((values[i] * values[j]) % p)
    return colors

def _try_additive(p, a, b):
    """Color edge (i,j) by (a*i + b*j) % p."""
    colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            colors.append((a * i + b * j) % p)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_colors = None
    best_count = 17
    
    # Base solution: (i + j) % 17
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    if _check_k4_fast(base_colors):
        best_colors = base_colors
        best_count = _num_colours(base_colors)
    
    # Strategy 1: Multiplicative colorings (v_i * v_j) % p
    # Use all nonzero residues mod p as vertex values
    primes_to_try = [17, 19, 23, 29, 31, 37, 41]
    
    for p in primes_to_try:
        if time.time() > deadline:
            break
        # Use values 1, 2, ..., 16 (or first 16 nonzero residues)
        values = list(range(1, 17))
        colors = _try_multiplicative(p, values)
        cnt = _num_colours(colors)
        if cnt < best_count:
            if _check_k4_fast(colors):
                best_colors = colors
                best_count = cnt
                if best_count <= 13:
                    return colors
        
        # Try different value sets: all nonzero residues
        if p > 16:
            values = list(range(1, p))[:16]
            colors = _try_multiplicative(p, values)
            cnt = _num_colours(colors)
            if cnt < best_count:
                if _check_k4_fast(colors):
                    best_colors = colors
                    best_count = cnt
                    if best_count <= 13:
                        return colors
    
    # Strategy 2: Additive colorings (a*i + b*j) % p
    primes_for_add = [13, 17, 19, 23, 29, 31]
    for p in primes_for_add:
        if time.time() > deadline:
            break
        for a in range(1, p):
            for b in range(1, p):
                if time.time() > deadline:
                    break
                colors = _try_additive(p, a, b)
                cnt = _num_colours(colors)
                if cnt < best_count:
                    if _check_k4_fast(colors):
                        best_colors = colors
                        best_count = cnt
                        if best_count <= 13:
                            return colors
    
    # Strategy 3: Mixed forms (a*i + b*j + c*i*j) % p
    for p in [17, 19, 23]:
        if time.time() > deadline:
            break
        for a in range(1, min(p, 6)):
            for b in range(1, min(p, 6)):
                for c in range(0, min(p, 4)):
                    if time.time() > deadline:
                        break
                    colors = []
                    for i in range(16):
                        for j in range(i + 1, 16):
                            colors.append((a * i + b * j + c * i * j) % p)
                    cnt = _num_colours(colors)
                    if cnt < best_count:
                        if _check_k4_fast(colors):
                            best_colors = colors
                            best_count = cnt
                            if best_count <= 13:
                                return colors
    
    # Strategy 4: Local improvement - try to reduce colors by remapping
    if best_colors is not None and time.time() < deadline - 20:
        current = list(best_colors)
        current_count = best_count
        
        # Try to merge colors: for each pair of colors, check if merging is valid
        for _ in range(50):  # limited iterations
            if time.time() > deadline - 5:
                break
            improved = False
            color_set = sorted(set(current))
            # Try merging the highest color into the lowest
            for target in range(len(color_set) - 1, 0, -1):
                for source in range(target - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    # Try remapping color_set[target] -> color_set[source]
                    new_colors = [color_set[source] if c == color_set[target] else c for c in current]
                    if _check_k4_fast(new_colors):
                        current = new_colors
                        current_count = _num_colours(current)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_count < best_count:
            best_colors = current
            best_count = current_count
    
    if best_colors is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_colors
# EVOLVE-BLOCK-END
