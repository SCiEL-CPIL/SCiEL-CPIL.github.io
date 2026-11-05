import time
import random
import itertools
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)
    
    num_k4s = len(k4s)
    
    # Precompute bitmask for each K4: which edges it contains
    # For fast distinct color counting, we'll track color masks per K4
    
    def count_distinct_colors(colors, k4_idx):
        """Count distinct colors in a K4 using bit operations."""
        e_indices = k4_edges[k4_idx]
        color_mask = 0
        for e in e_indices:
            c = colors[e]
            if c >= 0:
                color_mask |= (1 << c)
        return bin(color_mask).count('1')
    
    def is_valid_assignment_fast(colors, edge_idx, color):
        """
        Fast check if assigning 'color' to 'edge_idx' keeps all K4s valid.
        Only checks K4s that become fully colored or are already fully colored.
        """
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4_edges[k4_idx]
            # Check if this K4 is now fully colored
            all_colored = True
            for e in e_indices:
                if e == edge_idx:
                    continue
                if colors[e] < 0:
                    all_colored = False
                    break
            
            if not all_colored:
                # Not fully colored - optimistic: assume it can be fixed later
                # But check if it's already impossible: if 5+ edges are colored
                # and they have <= 4 distinct colors, the remaining edge(s) must add new colors
                # For now, be optimistic
                continue
            
            # Fully colored - check distinct count
            color_mask = (1 << color)
            for e in e_indices:
                if e != edge_idx:
                    color_mask |= (1 << colors[e])
            
            if bin(color_mask).count('1') < 5:
                return False
        
        return True
    
    def try_local_search(colors, target_C, max_iter=2000):
        """
        Local search: try to reduce the number of distinct colors used.
        Strategy: pick a color that appears least, try to recolor its edges to other colors.
        Uses 2-opt moves: if a move causes a violation, try to repair it by changing another edge in the violated K4.
        """
        if time.time() > deadline - 3:
            return colors
        
        current_colors = colors[:]
        
        # Count color frequencies
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Find the least used color
            color_counts = [0] * target_C
            for c in current_colors:
                if c >= 0:
                    color_counts[c] += 1
            
            # Find color with minimum count (but > 0)
            min_color = -1
            min_count = float('inf')
            for c in range(target_C):
                if 0 < color_counts[c] < min_count:
                    min_count = color_counts[c]
                    min_color = c
            
            if min_color < 0 or min_count == 0:
                break
            
            # Try to recolor edges with min_color to other colors
            edges_with_min = [e for e in range(num_edges) if current_colors[e] == min_color]
            random.shuffle(edges_with_min)
            
            improved = False
            for e_idx in edges_with_min:
                # Try each other color
                for new_c in range(target_C):
                    if new_c == min_color:
                        continue
                    
                    # Tentatively assign
                    current_colors[e_idx] = new_c
                    
                    # Check for violations
                    valid = True
                    violated_k4s = []
                    for k4_idx in edge_to_k4s[e_idx]:
                        e_indices = k4_edges[k4_idx]
                        color_mask = 0
                        for e in e_indices:
                            color_mask |= (1 << current_colors[e])
                        if bin(color_mask).count('1') < 5:
                            valid = False
                            violated_k4s.append(k4_idx)
                    
                    if valid:
                        # Check if this reduces total distinct colors
                        # We know min_color is being removed from e_idx
                        # If min_color is not used elsewhere, total count drops
                        # If it is used elsewhere, total count stays same, but we might enable further reductions
                        # For now, accept any valid move that removes min_color from e_idx
                        # The goal is to eventually eliminate min_color completely
                        improved = True
                        break
                    else:
                        # Try to repair violations
                        repaired = True
                        for k4_idx in violated_k4s:
                            e_indices = k4_edges[k4_idx]
                            current_mask = 0
                            for e in e_indices:
                                current_mask |= (1 << current_colors[e])
                            
                            # Find a color not in current_mask
                            repair_color = -1
                            for rc in range(target_C):
                                if not (current_mask & (1 << rc)):
                                    repair_color = rc
                                    break
                            
                            if repair_color == -1:
                                repaired = False
                                break
                            
                            # Pick an edge in this K4 to recolor
                            candidates = [e for e in e_indices if e != e_idx]
                            random.shuffle(candidates)
                            
                            success_repair = False
                            for e_repair in candidates:
                                old_repair_color = current_colors[e_repair]
                                current_colors[e_repair] = repair_color
                                
                                # Check if this repair causes new violations
                                check_ok = True
                                for k4_idx2 in edge_to_k4s[e_repair]:
                                    e_indices2 = k4_edges[k4_idx2]
                                    mask2 = 0
                                    for e in e_indices2:
                                        mask2 |= (1 << current_colors[e])
                                    if bin(mask2).count('1') < 5:
                                        check_ok = False
                                        break
                                
                                if check_ok:
                                    success_repair = True
                                    break
                                else:
                                    current_colors[e_repair] = old_repair_color
                            
                            if not success_repair:
                                repaired = False
                                break
                        
                        if repaired:
                            improved = True
                            break
                        else:
                            # Revert e_idx
                            current_colors[e_idx] = min_color
            
            if not improved:
                break
        
        return current_colors
    
    best_result = None
    best_colors_used = 100  # Start high
    
    # Baseline fallback: (i+j) % 17 gives at most 17 colors, but let's use a better one
    # Actually, let's construct a simple valid baseline
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Verify baseline
    valid = True
    for k4_idx in range(num_k4s):
        e_indices = k4_edges[k4_idx]
        k4_colors = [baseline[e] for e in e_indices]
        if len(set(k4_colors)) < 5:
            valid = False
            break
    if valid:
        best_result = baseline[:]
        best_colors_used = len(set(baseline))
    
    # Edge ordering heuristic: sort edges by number of K4s they belong to (descending)
    # Edges in more K4s are more constrained
    edge_degree = [len(edge_to_k4s[e]) for e in range(num_edges)]
    base_order = sorted(range(num_edges), key=lambda e: -edge_degree[e])
    
    for target_C in [13, 14, 15, 16]:
        if time.time() > deadline - 15:
            break
            
        attempts = 0
        max_attempts = 3000 if target_C == 13 else 8000
        
        while attempts < max_attempts:
            if time.time() > deadline - 5:
                break
                
            attempts += 1
            colors = [-1] * num_edges
            success = True
            
            # Use base order with some randomization
            edge_order = base_order[:]
            # Add some randomness: swap some adjacent elements
            for _ in range(10):
                pos = random.randint(0, num_edges - 2)
                edge_order[pos], edge_order[pos+1] = edge_order[pos+1], edge_order[pos]
            
            for e_idx in edge_order:
                # Try colors in random order
                color_options = list(range(target_C))
                random.shuffle(color_options)
                
                assigned = False
                for c in color_options:
                    if is_valid_assignment_fast(colors, e_idx, c):
                        colors[e_idx] = c
                        assigned = True
                        break
                
                if not assigned:
                    success = False
                    break
            
            if not success:
                continue
            
            # Verify the full coloring
            valid = True
            for k4_idx in range(num_k4s):
                e_indices = k4_edges[k4_idx]
                color_mask = 0
                for e in e_indices:
                    color_mask |= (1 << colors[e])
                if bin(color_mask).count('1') < 5:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Count distinct colors used
            num_used = len(set(colors))
            
            # Try local search to reduce colors
            refined = try_local_search(colors, target_C)
            refined_used = len(set(refined))
            
            if refined_used < best_colors_used:
                best_colors_used = refined_used
                best_result = refined[:]
                
                if best_colors_used <= 13:
                    return best_result
            elif num_used < best_colors_used:
                best_colors_used = num_used
                best_result = colors[:]
                
                if best_colors_used <= 13:
                    return best_result
    
    if best_result is not None:
        return best_result
    else:
        return baseline
