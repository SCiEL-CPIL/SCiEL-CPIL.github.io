import numpy as np
from itertools import combinations

def _compute_score(d_set, c_set):
    # d_set: tuple of 4 ints
    # c_set: tuple of 5 ints
    # Grid 1..9
    inc4 = 0
    inc3 = 0
    inc2 = 0
    
    # Precompute points for each d
    points_on_d = {}
    for d in d_set:
        pts = []
        for x in range(1, 10):
            y = x + d
            if 1 <= y <= 9:
                c = x + y
                pts.append((x, y, c))
        points_on_d[d] = pts
        
    # Precompute points for each c
    points_on_c = {}
    for c in c_set:
        pts = []
        for x in range(1, 10):
            y = c - x
            if 1 <= y <= 9:
                d = y - x
                pts.append((x, y, d))
        points_on_c[c] = pts
        
    # Count incidences for all 81 points
    # We need to know for each point if it's in d_set and c_set
    # Actually, we just need the distribution of (I_d, I_c)
    # I_d = 1 if y-x in d_set else 0
    # I_c = 1 if x+y in c_set else 0
    
    d_set_arr = set(d_set)
    c_set_arr = set(c_set)
    
    n4 = 0
    n3 = 0
    n2 = 0
    
    for x in range(1, 10):
        for y in range(1, 10):
            d = y - x
            c = x + y
            i_d = 1 if d in d_set_arr else 0
            i_c = 1 if c in c_set_arr else 0
            
            if i_d == 1 and i_c == 1:
                n4 += 1
            elif i_d == 1 or i_c == 1:
                n3 += 1
            else:
                n2 += 1
                
    # Select top 54
    # Take all n4
    remaining = 54
    score = 0
    if n4 >= remaining:
        return 4 * remaining
    score += 4 * n4
    remaining -= n4
    
    if n3 >= remaining:
        score += 3 * remaining
        return score
    score += 3 * n3
    remaining -= n3
    
    score += 2 * remaining
    return score

def szemeredi_trotter_incidences():
    # Search for best d_set and c_set
    best_score = -1
    best_d = ()
    best_c = ()
    
    possible_d = list(range(-8, 9))
    possible_c = list(range(2, 19))
    
    # To speed up, we can restrict search space or use random sampling
    # But 14M is okay.
    for d_comb in combinations(possible_d, 4):
        for c_comb in combinations(possible_c, 5):
            s = _compute_score(d_comb, c_comb)
            if s > best_score:
                best_score = s
                best_d = d_comb
                best_c = c_comb
                
    # Construct lines
    lines = []
    for i in range(1, 10):
        lines.append((1, 0, -i))
    for j in range(1, 10):
        lines.append((0, 1, -j))
    for d in best_d:
        lines.append((-1, 1, -d))
    for c in best_c:
        lines.append((1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    # Select points
    xs = np.arange(1, 10)
    ys = np.arange(1, 10)
    PX, PY = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    
    incidence_counts = np.zeros(len(all_points), dtype=np.int64)
    for idx in range(len(all_points)):
        x = all_points[idx, 0]
        y = all_points[idx, 1]
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                incidence_counts[idx] += 1
                
    sorted_indices = np.argsort(-incidence_counts)
    selected_indices = sorted_indices[:54]
    points = all_points[selected_indices]
    
    return points, lines
