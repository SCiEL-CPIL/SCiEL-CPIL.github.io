import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Precompute K4 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(k4_edges)
    
    # Edge index map
    edge_idx = {}
    for idx, e in enumerate(edges):
        edge_idx[e] = idx
        
    # Precompute K4 edge index arrays for fast access
    k4_edge_idxs = []
    for k4e in k4s:
        k4_edge_idxs.append([edge_idx[e] for e in k4e])
    
    # Precompute for each edge, which K4s contain it
    edge_k4s = [[] for _ in range(m)]
    for k4_idx, idxs in enumerate(k4_edge_idxs):
        for e_idx in idxs:
            edge_k4s[e_idx].append(k4_idx)

    def count_colors(coloring):
        return len(set(coloring.tolist()))
    
    def check_valid(coloring):
        """Check if every K4 has at least 5 distinct colors."""
        for idxs in k4_edge_idxs:
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    def count_violations(coloring):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for idxs in k4_edge_idxs:
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                violations += 1
        return violations

    def objective(coloring):
        """Objective: violations * 1000 + number of colors used."""
        v = count_violations(coloring)
        c = count_colors(coloring)
        return v * 1000 + c

    best_coloring = None
    best_colors = 120  # worst case
    best_valid = False
    
    random.seed(42)
    
    # Strategy 1: Try structured colorings based on vertex labels
    # color(i, j) = f(label[i], label[j], mod)
    
    def func_sum(l1, l2, mod): return (l1 + l2) % mod
    def func_xor(l1, l2, mod): return (l1 ^ l2) % mod
    def func_diff(l1, l2, mod): return (l1 - l2) % mod
    def func_prod(l1, l2, mod): return (l1 * l2) % mod
    def func_sum2(l1, l2, mod): return (l1 + 2*l2) % mod
    def func_diff2(l1, l2, mod): return (l1 - 2*l2) % mod
    
    functions = [func_sum, func_xor, func_diff, func_prod, func_sum2, func_diff2]
    
    # We prioritize smaller moduli to minimize color count
    # Target: find valid coloring with mod 13, 14, 15
    for mod in [13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 15:
            break
            
        for f in functions:
            if time.time() > deadline - 15:
                break
                
            # Try multiple random labelings
            max_trials = 300 if mod <= 15 else 150
            for trial in range(max_trials):
                if time.time() > deadline - 10:
                    break
                    
                # Assign random labels to vertices
                labels = [random.randint(0, mod - 1) for _ in range(n)]
                
                # Derive edge colors
                coloring = np.zeros(m, dtype=int)
                for idx, (i, j) in enumerate(edges):
                    coloring[idx] = f(labels[i], labels[j], mod)
                
                # Check validity
                if check_valid(coloring):
                    n_colors = count_colors(coloring)
                    if n_colors < best_colors:
                        best_colors = n_colors
                        best_coloring = coloring.copy()
                        best_valid = True
                        
                        # If we found 13 colors, that's optimal (lower bound)
                        if best_colors <= 13:
                            # Do a quick polish and return
                            break
                else:
                    # Local search: try to fix violations by changing one vertex label
                    # Proper local search with revert
                    for ls_step in range(30):
                        if time.time() > deadline - 10:
                            break
                        v = random.randint(0, n - 1)
                        old_label = labels[v]
                        new_label = random.randint(0, mod - 1)
                        if new_label == old_label:
                            continue
                        
                        # Save old colors for incident edges
                        incident_edges = [(idx, i, j) for idx, (i, j) in enumerate(edges) if i == v or j == v]
                        old_colors = {idx: coloring[idx] for idx, _, _ in incident_edges}
                        
                        # Apply new label
                        labels[v] = new_label
                        for idx, i, j in incident_edges:
                            coloring[idx] = f(labels[i], labels[j], mod)
                        
                        # Check if improved
                        if check_valid(coloring):
                            n_colors = count_colors(coloring)
                            if n_colors < best_colors:
                                best_colors = n_colors
                                best_coloring = coloring.copy()
                                best_valid = True
                                if best_colors <= 13:
                                    break
                        else:
                            # Revert
                            labels[v] = old_label
                            for idx, i, j in incident_edges:
                                coloring[idx] = old_colors[idx]
                        
                        if best_colors <= 13:
                            break
                        
                        if ls_step == 29 and best_colors > 13:
                            # Try one more round with different strategy
                            pass

            if best_colors <= 13:
                break

    # Strategy 2: Try direct edge-coloring with small palette
    # Assign colors directly to edges, using local search
    if time.time() < deadline - 20:
        for palette_size in [13, 14, 15]:
            if time.time() > deadline - 10:
                break
            
            # Initialize with random coloring using palette_size colors
            coloring = np.random.randint(0, palette_size, size=m)
            
            # Local search: minimize violations, then minimize colors used
            current_obj = objective(coloring)
            no_improve = 0
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random edge
                e_idx = random.randint(0, m - 1)
                old_color = coloring[e_idx]
                
                # Try all colors in palette
                best_new_color = old_color
                best_new_obj = current_obj
                
                for c in range(palette_size):
                    if c == old_color:
                        continue
                    coloring[e_idx] = c
                    new_obj = objective(coloring)
                    if new_obj < best_new_obj:
                        best_new_obj = new_obj
                        best_new_color = c
                    coloring[e_idx] = old_color
                
                if best_new_color != old_color:
                    coloring[e_idx] = best_new_color
                    current_obj = best_new_obj
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 500:
                        break
                
                # Check if we found a valid solution
                if current_obj < palette_size:  # violations = 0, colors <= palette_size
                    n_colors = count_colors(coloring)
                    if n_colors < best_colors:
                        best_colors = n_colors
                        best_coloring = coloring.copy()
                        best_valid = True
                        if best_colors <= 13:
                            break
            
            if best_colors <= 13:
                break

    # If no valid solution found, use fallback
    if best_coloring is None or not best_valid:
        # Try (i+j) % 17
        coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=int)
        if check_valid(coloring):
            best_coloring = coloring
            best_colors = count_colors(coloring)
            best_valid = True
        else:
            # Last resort: distinct colors for all edges
            coloring = np.arange(m, dtype=int)
            best_coloring = coloring
            best_colors = m
            best_valid = True

    # Final Polish: Try to reduce colors on the best coloring found
    if best_colors > 13 and time.time() < deadline - 5:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        # Local search: try to recolor an edge to reduce unique color count
        for polish_iter in range(3000):
            if time.time() > deadline - 3:
                break
            
            # Find colors that appear exactly once
            color_counts = {}
            for c in coloring:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            singleton_colors = [c for c, count in color_counts.items() if count == 1]
            
            if not singleton_colors:
                break
            
            target_color = random.choice(singleton_colors)
            target_idx = np.where(coloring == target_color)[0][0]
            
            # Try to change it to another existing color
            other_colors = [c for c in color_counts.keys() if c != target_color]
            random.shuffle(other_colors)
            
            found_better = False
            for new_color in other_colors:
                old_color = coloring[target_idx]
                coloring[target_idx] = new_color
                
                if check_valid(coloring):
                    new_count = count_colors(coloring)
                    if new_count < current_colors:
                        current_colors = new_count
                        best_colors = new_count
                        best_coloring = coloring.copy()
                        found_better = True
                        break
                    else:
                        coloring[target_idx] = old_color
                        break
                else:
                    coloring[target_idx] = old_color
                    # Try next color
            
            if not found_better and len(singleton_colors) == 1:
                # Only one singleton and we couldn't merge it; try random edge recoloring
                e_idx = random.randint(0, m - 1)
                old_color = coloring[e_idx]
                for c in range(120):
                    coloring[e_idx] = c
                    if check_valid(coloring):
                        new_count = count_colors(coloring)
                        if new_count < current_colors:
                            current_colors = new_count
                            best_colors = new_count
                            best_coloring = coloring.copy()
                            break
                        else:
                            coloring[e_idx] = old_color
                            break
                    else:
                        coloring[e_idx] = old_color

    # Return the best coloring found
    result = [int(c) for c in best_coloring]
    # Remap colors to 0..K-1
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
