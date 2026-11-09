import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Generate E8 roots (scaled by 2)
    # Type 1: Two coords +/-2, rest 0
    e8_roots = []
    for i in range(8):
        for j in range(8):
            if i == j: continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_roots.append(v)
    
    # Type 2: All coords +/-1, even number of -1s
    for mask in range(256):
        # Check even number of -1s
        # We can represent the sign vector
        signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(8)])
        # Count -1s
        if np.sum(signs == -1) % 2 == 0:
            e8_roots.append(signs)
            
    e8_roots = np.array(e8_roots, dtype=np.int64)
    
    # Generate D3 roots (scaled by 2)
    # D3 roots are +/-ei +/-ej
    d3_roots = []
    for i in range(3):
        for j in range(3):
            if i == j: continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    d3_roots.append(v)
                    
    d3_roots = np.array(d3_roots, dtype=np.int64)
    
    # Find max subset of d3_roots with pairwise dist squared >= 16
    n_d3 = len(d3_roots)
    # Precompute distance matrix
    dists = np.zeros((n_d3, n_d3), dtype=np.int64)
    for i in range(n_d3):
        for j in range(n_d3):
            diff = d3_roots[i] - d3_roots[j]
            dists[i, j] = np.dot(diff, diff)
            
    best_subset = []
    # Brute force all subsets? 2^12 = 4096. Easy.
    for r in range(n_d3, 0, -1):
        found = False
        for combo in itertools.combinations(range(n_d3), r):
            valid = True
            for i in range(len(combo)):
                for j in range(i+1, len(combo)):
                    if dists[combo[i], combo[j]] < 16:
                        valid = False
                        break
                if not valid: break
            if valid:
                best_subset = list(combo)
                found = True
                break
        if found:
            break
            
    v_selected = d3_roots[best_subset]
    
    # Construct points
    # Cartesian product of e8_roots and v_selected
    # Shape (240 * len(v_selected), 11)
    n_pts = len(e8_roots) * len(v_selected)
    points = np.zeros((n_pts, 11), dtype=np.int64)
    
    idx = 0
    for u in e8_roots:
        for v in v_selected:
            points[idx, :8] = u
            points[idx, 8:] = v
            idx += 1
            
    return points
