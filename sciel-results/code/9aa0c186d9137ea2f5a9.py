import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Generate all K4s and map edges to K4 indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(k4_edges)
    
    num_k4s = len(k4s)  # 1820
    
    # Edge index map
    edge_idx = {}
    for idx, e in enumerate(edges):
        edge_idx[e] = idx
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(m)]
    for k4_idx, k4e in enumerate(k4s):
        for e in k4e:
            edge_to_k4s[edge_idx[e]].append(k4_idx)
    
    # Precompute K4 edge indices
    k4_edge_indices = []
    for k4e in k4s:
        k4_edge_indices.append([edge_idx[e] for e in k4e])
    
    def count_colors(coloring):
        return len(set(coloring))
    
    def count_violations(coloring):
        violations = 0
        for k4_idx in range(num_k4s):
            idxs = k4_edge_indices[k4_idx]
            colors = set(coloring[i] for i in idxs)
            if len(colors) < 5:
                violations += 1
        return violations
    
    def count_violations_for_k4s(coloring, k4_indices):
        violations = 0
        for k4_idx in k4_indices:
            idxs = k4_edge_indices[k4_idx]
            colors = set(coloring[i] for i in idxs)
            if len(colors) < 5:
                violations += 1
        return violations
    
    def check_valid(coloring):
        return count_violations(coloring) == 0
    
    random.seed(42)
    
    # Strategy: Start with a greedy construction that tries to use few colors,
    # then use simulated annealing to reduce colors further.
    
    # First, try a greedy approach with a small color budget
    def greedy_construct(max_colors):
        """Try to construct a valid coloring using at most max_colors colors."""
        coloring = np.full(m, -1, dtype=int)
        
        # Process edges in order of most constrained first (edges in most K4s)
        edge_order = sorted(range(m), key=lambda i: len(edge_to_k4s[i]), reverse=True)
        
        for idx in edge_order:
            # Try each color
            valid_colors = []
            for c in range(max_colors):
                coloring[idx] = c
                # Check if this creates any violation in K4s containing this edge
                # that are now fully colored
                ok = True
                for k4_idx in edge_to_k4s[idx]:
                    idxs = k4_edge_indices[k4_idx]
                    # Check if all edges in this K4 are colored
                    if all(coloring[j] >= 0 for j in idxs):
                        colors = set(coloring[j] for j in idxs)
                        if len(colors) < 5:
                            ok = False
                            break
                if ok:
                    valid_colors.append(c)
            
            if valid_colors:
                coloring[idx] = random.choice(valid_colors)
            else:
                coloring[idx] = -1
        
        # Check if all edges are colored
        if np.any(coloring < 0):
            return None, -1
        
        # Check validity
        if not check_valid(coloring):
            return None, -1
        
        return coloring, count_colors(coloring)
    
    # Try greedy with small color budgets
    best_coloring = None
    best_colors = 18  # start with a high value
    
    for max_c in range(13, 18):
        coloring, nc = greedy_construct(max_c)
        if coloring is not None and nc < best_colors:
            best_colors = nc
            best_coloring = coloring.copy()
        if time.time() > deadline - 30:
            break
    
    # If greedy failed, start with the seed
    if best_coloring is None:
        best_coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=int)
        best_colors = count_colors(best_coloring)
    
    # Simulated annealing to reduce colors
    coloring = best_coloring.copy()
    current_colors = best_colors
    current_violations = count_violations(coloring)
    
    T = 10.0
    cooling = 0.999
    max_iters = 100000
    iter_count = 0
    
    while iter_count < max_iters:
        if time.time() > deadline - 5:
            break
        
        iter_count += 1
        
        # Pick a random edge
        idx = random.randint(0, m - 1)
        old_color = coloring[idx]
        
        # Try a new color (possibly introducing a new color or removing one)
        max_possible = max(current_colors, 15)
        new_color = random.randint(0, max_possible - 1)
        
        if new_color == old_color:
            continue
        
        coloring[idx] = new_color
        
        # Compute change in violations for affected K4s
        affected_k4s = edge_to_k4s[idx]
        
        # We need to check violations before and after
        # For efficiency, only check affected K4s
        old_violations = 0
        for k4_idx in affected_k4s:
            idxs = k4_edge_indices[k4_idx]
            # Temporarily restore old color to count
            coloring[idx] = old_color
            colors = set(coloring[j] for j in idxs)
            if len(colors) < 5:
                old_violations += 1
            coloring[idx] = new_color
        
        new_violations = 0
        for k4_idx in affected_k4s:
            idxs = k4_edge_indices[k4_idx]
            colors = set(coloring[j] for j in idxs)
            if len(colors) < 5:
                new_violations += 1
        
        delta_violations = new_violations - old_violations
        
        # Compute change in color count
        new_color_count = count_colors(coloring)
        delta_colors = new_color_count - current_colors
        
        # Acceptance criterion:
        # - If violations decrease or stay same, and colors don't increase too much, accept
        # - If colors decrease, accept even if violations increase slightly
        # - Use simulated annealing for violations
        
        if delta_violations <= 0 and delta_colors <= 0:
            # Both improve or stay same
            accept = True
        elif delta_violations <= 0 and delta_colors > 0:
            # Violations improve but colors increase - accept with some probability
            accept = random.random() < np.exp(-delta_colors / T)
        elif delta_violations > 0 and delta_colors < 0:
            # Colors improve but violations increase - accept with some probability
            accept = random.random() < np.exp(-delta_violations / T)
        elif delta_violations > 0 and delta_colors > 0:
            # Both worsen - rarely accept
            accept = random.random() < np.exp(-(delta_violations + delta_colors) / T)
        else:
            accept = False
        
        if accept:
            current_violations += delta_violations
            if new_color_count < best_colors:
                best_colors = new_color_count
                best_coloring = coloring.copy()
            current_colors = new_color_count
        else:
            coloring[idx] = old_color
        
        T *= cooling
    
    # Final cleanup: try to reduce colors by recoloring
    for _ in range(10000):
        if time.time() > deadline - 2:
            break
        
        # Find the highest color in use
        max_color = coloring.max()
        max_edges = np.where(coloring == max_color)[0]
        
        if len(max_edges) == 0:
            break
        
        idx = random.choice(max_edges)
        old_color = coloring[idx]
        
        # Try to recolor to a lower color
        for new_color in range(max_color):
            coloring[idx] = new_color
            affected_k4s = edge_to_k4s[idx]
            
            # Check if valid
            ok = True
            for k4_idx in affected_k4s:
                idxs = k4_edge_indices[k4_idx]
                colors = set(coloring[j] for j in idxs)
                if len(colors) < 5:
                    ok = False
                    break
            
            if ok:
                new_count = count_colors(coloring)
                if new_count < best_colors:
                    best_colors = new_count
                    best_coloring = coloring.copy()
                break
            else:
                coloring[idx] = old_color
    
    # Remap colors to be compact
    result = [int(c) for c in best_coloring]
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
