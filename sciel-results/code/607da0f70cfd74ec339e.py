import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    n = 16
    # Precompute all K4s and their edge indices
    # Edge (i,j) for i<j has index: sum_{k=0}^{i-1} (n-1-k) + (j-i-1) = i*(2n-1-i)//2 + j - i - 1
    # Or simpler: build a list
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
    
    def count_violations(colours):
        """Return number of K4s with fewer than 5 distinct colors."""
        violations = 0
        for k4 in k4s:
            colors_in_k4 = set()
            for eidx in k4:
                colors_in_k4.add(colours[eidx])
            if len(colors_in_k4) < 5:
                violations += 1
        return violations
    
    def is_valid(colours):
        return count_violations(colours) == 0
    
    # Start with seed (i+j) % 17
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
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
                
                # Create new coloring with c1 -> c2
                new_colours = list(current)
                for k in range(len(new_colours)):
                    if new_colours[k] == c1:
                        new_colours[k] = c2
                
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
                
                random.seed(restart * 1000 + target_colors)
                colours = [random.randint(0, target_colors - 1) for _ in range(120)]
                
                violations = count_violations(colours)
                if violations == 0:
                    best_colours = colours
                    best_violations = 0
                    break
                
                # Local search: flip edges to reduce violations
                for step in range(1000):
                    if time.time() > deadline:
                        break
                    
                    if violations == 0:
                        best_colours = colours
                        best_violations = 0
                        break
                    
                    # Pick a random edge and try changing its color
                    eidx = random.randint(0, 119)
                    old_color = colours[eidx]
                    new_color = random.randint(0, target_colors - 1)
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
    
    return current
