"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    n = 16
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Build edge index map
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
    k4_edge_indices = []
    for k4 in all_k4s:
        eidxs = [edge_to_idx[(a, b)] for a, b in itertools.combinations(k4, 2)]
        k4_edge_indices.append(eidxs)
    
    # Precompute for each edge, the list of K4s containing it
    edge_k4s = [[] for _ in range(num_edges)]
    for ki, k4 in enumerate(all_k4s):
        for eidx in k4_edge_indices[ki]:
            edge_k4s[eidx].append(ki)
    
    # Precompute for each K4, the other 5 edges for each of its 6 edges
    k4_other_edges = []
    k4_edge_pos = []  # position of each edge within its K4
    for ki, eidxs in enumerate(k4_edge_indices):
        others = []
        pos = []
        for e in eidxs:
            others.append([x for x in eidxs if x != e])
            pos.append(eidxs.index(e))
        k4_other_edges.append(others)
        k4_edge_pos.append(pos)
    
    def is_valid(colors):
        for ki in range(len(all_k4s)):
            eidxs = k4_edge_indices[ki]
            cset = set(colors[e] for e in eidxs)
            if len(cset) < 5:
                return False
        return True
    
    def count_violations(colors):
        """Count K4s with fewer than 5 distinct colors."""
        count = 0
        for ki in range(len(all_k4s)):
            eidxs = k4_edge_indices[ki]
            cset = set(colors[e] for e in eidxs)
            if len(cset) < 5:
                count += 1
        return count
    
    def try_eliminate_color(target_color, colors, used_colors, color_counts):
        """Try to recolor all edges of target_color to other colors. Returns True if eliminated."""
        edges_with_color = [idx for idx, c in enumerate(colors) if c == target_color]
        if not edges_with_color:
            used_colors.discard(target_color)
            return True
        
        # Sort edges by number of affected K4s (ascending) to try easier ones first
        edges_with_color.sort(key=lambda e: len(edge_k4s[e]))
        
        for idx in edges_with_color:
            affected_k4s = edge_k4s[idx]
            
            # Find valid new colors
            valid_new_colors = []
            for new_color in used_colors:
                if new_color == target_color:
                    continue
                
                valid = True
                for ki in affected_k4s:
                    other_eidxs = k4_other_edges[ki][k4_edge_pos[ki][k4_edge_indices[ki].index(idx)]]
                    other_colors = set(colors[e] for e in other_eidxs)
                    if new_color in other_colors:
                        if len(other_colors) < 5:
                            valid = False
                            break
                    else:
                        if len(other_colors) < 4:
                            valid = False
                            break
                
                if valid:
                    valid_new_colors.append(new_color)
            
            if valid_new_colors:
                valid_new_colors.sort(key=lambda c: -color_counts.get(c, 0))
                new_color = valid_new_colors[0]
                colors[idx] = new_color
                color_counts[target_color] = color_counts.get(target_color, 0) - 1
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
            else:
                return False
        
        used_colors.discard(target_color)
        if color_counts.get(target_color, 0) == 0:
            del color_counts[target_color]
        return True
    
    def local_search_swap(colors, used_colors, color_counts, max_iters=5000):
        """Try to enable further color elimination via swaps."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try to eliminate the least-used color
            target_color = min(used_colors, key=lambda c: color_counts.get(c, 0))
            
            if try_eliminate_color(target_color, colors, used_colors, color_counts):
                improved = True
                continue
            
            # Try a swap: find two edges with different colors where swapping helps
            edges_with_target = [idx for idx, c in enumerate(colors) if c == target_color]
            if not edges_with_target:
                continue
            
            # Try swapping target_color edges with other colors
            for idx1 in edges_with_target:
                c1 = colors[idx1]
                # Find edges with other colors that could swap
                for idx2 in range(num_edges):
                    if idx2 == idx1:
                        continue
                    c2 = colors[idx2]
                    if c2 == c1:
                        continue
                    
                    # Check if swapping c1 and c2 would be valid
                    # Temporarily swap
                    colors[idx1] = c2
                    colors[idx2] = c1
                    
                    # Check validity
                    valid = True
                    for ki in edge_k4s[idx1]:
                        eidxs = k4_edge_indices[ki]
                        cset = set(colors[e] for e in eidxs)
                        if len(cset) < 5:
                            valid = False
                            break
                    if valid:
                        for ki in edge_k4s[idx2]:
                            eidxs = k4_edge_indices[ki]
                            cset = set(colors[e] for e in eidxs)
                            if len(cset) < 5:
                                valid = False
                                break
                    
                    if valid:
                        # Accept the swap
                        color_counts[c1] = color_counts.get(c1, 0) - 1
                        color_counts[c2] = color_counts.get(c2, 0) - 1
                        color_counts[c2] = color_counts.get(c2, 0) + 1
                        color_counts[c1] = color_counts.get(c1, 0) + 1
                        improved = True
                        break
                    else:
                        # Revert
                        colors[idx1] = c1
                        colors[idx2] = c2
                
                if improved:
                    break
        
        return colors
    
    # Generate multiple initial colorings
    initial_colorings = []
    
    # 1. (i+j) mod k for various k
    for k in [13, 14, 15, 16, 17, 18, 19]:
        colors = [(i + j) % k for i in range(n) for j in range(i + 1, n)]
        initial_colorings.append(colors)
    
    # 2. (j-i) mod k
    for k in [13, 14, 15, 16, 17]:
        colors = [(j - i) % k for i in range(n) for j in range(i + 1, n)]
        initial_colorings.append(colors)
    
    # 3. Random colorings with limited palette
    random.seed(42)
    for k in [13, 14, 15, 16]:
        for _ in range(5):
            colors = [random.randint(0, k - 1) for _ in range(num_edges)]
            initial_colorings.append(colors)
    
    best_colors = None
    best_num_colors = float('inf')
    
    for init_colors in initial_colorings:
        if time.time() >= deadline:
            break
        
        # Check if valid; if not, try to fix
        colors = init_colors[:]
        
        if not is_valid(colors):
            # Try to fix by adding colors (shouldn't happen for sum mod k >= 17)
            # For now, skip invalid colorings
            continue
        
        used_colors = set(colors)
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Run elimination
        max_iterations = 50000
        iteration = 0
        stuck_count = 0
        
        while iteration < max_iterations and time.time() < deadline:
            iteration += 1
            
            current_num = len(used_colors)
            if current_num < best_num_colors:
                best_colors = colors[:]
                best_num_colors = current_num
            
            if best_num_colors <= 13:
                break
            
            # Pick target: least-used color
            target_color = min(used_colors, key=lambda c: color_counts.get(c, 0))
            
            success = try_eliminate_color(target_color, colors, used_colors, color_counts)
            
            if success:
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > 30:
                    stuck_count = 0
                    # Try random color
                    target_color = random.choice(list(used_colors))
                    success = try_eliminate_color(target_color, colors, used_colors, color_counts)
                    if success:
                        stuck_count = 0
        
        # Final check
        if is_valid(colors):
            final_num = len(set(colors))
            if final_num < best_num_colors:
                best_colors = colors[:]
                best_num_colors = final_num
    
    # Fallback: use the classic (i+j) % 17
    if best_colors is None:
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        best_num_colors = 17
    
    return best_colors
# EVOLVE-BLOCK-END
