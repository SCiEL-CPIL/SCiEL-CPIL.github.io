"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    n = 16
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    # Build edge index map: (i,j) -> index in colors list
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    num_edges = idx  # 120
    
    # Precompute all K4s and their 6 edge indices
    all_k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(all_k4s)
    k4_edge_indices = []
    for k4 in all_k4s:
        eidxs = [edge_to_idx[(a, b)] for a, b in itertools.combinations(k4, 2)]
        k4_edge_indices.append(eidxs)
    
    # Precompute for each edge, the list of K4 indices containing it
    edge_k4s = [[] for _ in range(num_edges)]
    for ki in range(num_k4s):
        for eidx in k4_edge_indices[ki]:
            edge_k4s[eidx].append(ki)
    
    # Precompute for each K4 and each of its 6 edges, the other 5 edge indices
    k4_other_edges = []
    for ki, eidxs in enumerate(k4_edge_indices):
        others = []
        for e in eidxs:
            others.append([x for x in eidxs if x != e])
        k4_other_edges.append(others)
    
    # Initial coloring: (i+j) mod 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    def count_k4_colors(ki, colors):
        eidxs = k4_edge_indices[ki]
        cset = set(colors[e] for e in eidxs)
        return len(cset)
    
    def is_valid(colors):
        for ki in range(num_k4s):
            eidxs = k4_edge_indices[ki]
            cset = set(colors[e] for e in eidxs)
            if len(cset) < 5:
                return False
        return True
    
    # Check initial validity
    if not is_valid(colors):
        return colors
    
    used_colors = set(colors)
    
    random.seed(42)
    
    def try_eliminate_color(target_color):
        """Try to recolor all edges of target_color to other colors. Returns True if eliminated."""
        nonlocal used_colors, colors, best_colors, best_num_colors
        
        edges_with_color = [idx for idx, c in enumerate(colors) if c == target_color]
        if not edges_with_color:
            used_colors.discard(target_color)
            return True
        
        # Sort edges by number of affected K4s (ascending) to try easier ones first
        edges_with_color.sort(key=lambda e: len(edge_k4s[e]))
        
        for idx in edges_with_color:
            affected_k4s = edge_k4s[idx]
            
            # Find valid new colors: for each candidate color, check all affected K4s
            valid_new_colors = []
            for new_color in used_colors:
                if new_color == target_color:
                    continue
                
                valid = True
                for ki in affected_k4s:
                    # Get the 5 other edges in this K4
                    other_eidxs = k4_other_edges[ki][k4_edge_indices[ki].index(idx)]
                    # Count distinct colors among other 5 edges
                    other_colors = set(colors[e] for e in other_eidxs)
                    
                    # After recoloring this edge to new_color, the K4 has:
                    # - new_color for this edge
                    # - colors from other 5 edges
                    # Total distinct colors = len(other_colors) + (1 if new_color not in other_colors else 0)
                    total_colors = len(other_colors) + (1 if new_color not in other_colors else 0)
                    
                    if total_colors < 5:
                        valid = False
                        break
                
                if valid:
                    valid_new_colors.append(new_color)
            
            if valid_new_colors:
                # Pick a valid new color - prefer one that appears most (to consolidate)
                color_counts = {}
                for c in colors:
                    color_counts[c] = color_counts.get(c, 0) + 1
                valid_new_colors.sort(key=lambda c: -color_counts.get(c, 0))
                new_color = valid_new_colors[0]
                colors[idx] = new_color
                
                # Check if target_color is now fully eliminated
                if all(colors[e] != target_color for e in edges_with_color):
                    used_colors.discard(target_color)
                    current_num = len(used_colors)
                    if current_num < best_num_colors:
                        best_colors = colors[:]
                        best_num_colors = current_num
                    return True
            else:
                # Cannot recolor this edge to any valid color
                return False
        
        return True
    
    # Main search loop
    iteration = 0
    max_iterations = 100000
    
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        
        current_colors = set(colors)
        if len(current_colors) < best_num_colors:
            best_colors = colors[:]
            best_num_colors = len(current_colors)
        
        if len(used_colors) <= 13:
            break
        
        # Pick a random color to try to eliminate - prefer less frequent colors
        color_freq = {}
        for c in colors:
            color_freq[c] = color_freq.get(c, 0) + 1
        
        # Sort colors by frequency (ascending) and pick from the less frequent ones
        sorted_colors = sorted(used_colors, key=lambda c: color_freq.get(c, 0))
        
        # Try the 3 least frequent colors first
        candidates = sorted_colors[:min(3, len(sorted_colors))]
        target_color = random.choice(candidates)
        
        if try_eliminate_color(target_color):
            continue
        
        # If elimination failed, try a random swap to escape local optima
        if time.time() < deadline - 5.0:
            # Pick two random edges with different colors and swap
            e1 = random.randint(0, num_edges - 1)
            e2 = random.randint(0, num_edges - 1)
            if e1 != e2 and colors[e1] != colors[e2]:
                old_c1, old_c2 = colors[e1], colors[e2]
                colors[e1], colors[e2] = old_c2, old_c1
                
                # Check validity after swap
                valid = True
                for ki in edge_k4s[e1]:
                    if count_k4_colors(ki, colors) < 5:
                        valid = False
                        break
                if valid:
                    for ki in edge_k4s[e2]:
                        if count_k4_colors(ki, colors) < 5:
                            valid = False
                            break
                
                if not valid:
                    # Revert the swap
                    colors[e1], colors[e2] = old_c1, old_c2
    
    return best_colors
# EVOLVE-BLOCK-END
