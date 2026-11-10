import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    n = 16
    # Precompute all K4s and their edge indices
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))
    
    # Precompute numpy matrix for K4s
    k4_matrix = np.array(k4s, dtype=np.int32)

    def count_violations(colours):
        """Return number of K4s with fewer than 5 distinct colors."""
        if not isinstance(colours, np.ndarray):
            colours = np.array(colours)
        k4_colors = colours[k4_matrix]
        sorted_rows = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_rows, axis=1)
        is_new = (diffs != 0)
        num_distinct = 1 + is_new.sum(axis=1)
        return int((num_distinct < 5).sum())

    def is_valid(colours):
        return count_violations(colours) == 0

    # Start with seed (i+j) % 17
    current = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    # Try to reduce colors by merging
    current_set = set(current)
    
    # Greedy merge: try to merge color pairs
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current_set = set(current)
        if len(current_set) <= 13:
            break
        
        sorted_colors = sorted(current_set)
        merged = False
        
        # Try all pairs of colors to merge
        for i in range(len(sorted_colors)):
            for j in range(i + 1, len(sorted_colors)):
                if time.time() > deadline:
                    break
                
                c1 = sorted_colors[i]
                c2 = sorted_colors[j]
                
                new_colours = current.copy()
                new_colours[new_colours == c1] = c2
                
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            break
    
    # If still above 13, try local search with random restarts
    if len(set(current)) > 13:
        best_colours = current
        best_violations = count_violations(current)
        
        # Try to find a valid coloring with fewer colors
        for target_colors in range(len(set(current)) - 1, 12, -1):
            if time.time() > deadline:
                break
            
            # Random initialization with target_colors colors
            for restart in range(50):
                if time.time() > deadline:
                    break
                
                colours = np.random.randint(0, target_colors, 120)
                
                violations = count_violations(colours)
                if violations == 0:
                    best_colours = colours
                    best_violations = 0
                    break
                
                # Local search: flip edges to reduce violations
                for step in range(3000):
                    if time.time() > deadline:
                        break
                    
                    if violations == 0:
                        best_colours = colours
                        best_violations = 0
                        break
                    
                    eidx = np.random.randint(0, 120)
                    old_color = colours[eidx]
                    new_color = np.random.randint(0, target_colors)
                    if new_color == old_color:
                        continue
                    
                    colours[eidx] = new_color
                    new_violations = count_violations(colours)
                    
                    if new_violations <= violations:
                        violations = new_violations
                    else:
                        colours[eidx] = old_color
        
        if best_violations == 0:
            current = best_colours
    
    return current.tolist()
