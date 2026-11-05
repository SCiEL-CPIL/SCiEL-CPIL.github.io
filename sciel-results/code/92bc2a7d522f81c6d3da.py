"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s
    
    n = 16
    # Generate all edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    num_edges = len(edges)  # 120
    
    # Create edge index lookup
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
    
    # Generate all K4s (sets of 4 vertices)
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4s.append((a, b, c, d))
    
    num_k4s = len(k4s)
    
    # For each K4, precompute the 6 edges it contains
    k4_edges = []
    for (a, b, c, d) in k4s:
        e_list = [
            edge_index[(a, b)],
            edge_index[(a, c)],
            edge_index[(a, d)],
            edge_index[(b, c)],
            edge_index[(b, d)],
            edge_index[(c, d)]
        ]
        k4_edges.append(e_list)
    
    # For each edge, precompute which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
    
    def count_distinct_colors_in_k4(k4_idx, colors):
        e_list = k4_edges[k4_idx]
        seen = set()
        for e_idx in e_list:
            seen.add(colors[e_idx])
        return len(seen)
    
    def check_validity(colors):
        for k4_idx in range(num_k4s):
            if count_distinct_colors_in_k4(k4_idx, colors) < 5:
                return False
        return True
    
    def get_max_color_used(colors):
        return int(max(colors)) + 1
    
    # Initial coloring: (i + j) mod 17
    colors = np.zeros(num_edges, dtype=np.int32)
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
    
    best_colors = colors.copy()
    best_max_color = get_max_color_used(best_colors)
    
    # Verify initial is valid
    assert check_validity(best_colors)
    
    def try_improve(colors, max_allowed_colors):
        """Try to reduce max color used below max_allowed_colors using local search."""
        current = colors.copy()
        max_color = max_allowed_colors - 1
        
        # First, try to recolor all edges to use at most max_color colors
        # Greedy: for each edge, try to assign a color in [0, max_color] that maintains validity
        for attempt in range(100):
            # Reset and try greedy assignment
            current = np.zeros(num_edges, dtype=np.int32)
            assigned = np.zeros(num_edges, dtype=bool)
            
            # Random order for greedy
            order = np.random.permutation(num_edges)
            
            success = True
            for e_idx in order:
                best_c = -1
                # Try colors in random order within [0, max_color]
                c_order = np.random.permutation(max_color + 1)
                for c in c_order:
                    current[e_idx] = c
                    # Check if this creates a violation in any K4 containing e_idx
                    valid = True
                    for k4_idx in edge_to_k4s[e_idx]:
                        # Only check K4s where all edges are assigned
                        all_assigned = all(assigned[e2] for e2 in k4_edges[k4_idx])
                        if not all_assigned:
                            continue
                        if count_distinct_colors_in_k4(k4_idx, current) < 5:
                            valid = False
                            break
                    if valid:
                        best_c = c
                        break
                if best_c == -1:
                    success = False
                    break
                current[e_idx] = best_c
                assigned[e_idx] = True
            
            if success and check_validity(current):
                return current
        
        return None
    
    # Local search to reduce max color
    for target_max in range(14, 8, -1):  # Try 14, 13, ..., 9
        if time.time() > deadline - 5.0:
            break
        result = try_improve(best_colors, target_max)
        if result is not None:
            best_colors = result
            best_max_color = get_max_color_used(best_colors)
    
    # Simulated annealing to further optimize
    temp = 1.0
    cool_rate = 0.999
    current = best_colors.copy()
    current_max = get_max_color_used(current)
    
    # Precompute K4 distinct counts
    k4_distinct = np.array([count_distinct_colors_in_k4(i, current) for i in range(num_k4s)])
    
    iterations = 0
    max_iterations = 500000
    
    while time.time() < deadline - 2.0 and iterations < max_iterations:
        iterations += 1
        
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = current[e_idx]
        
        # Pick a new color
        # Bias toward lower colors
        max_c = min(current_max - 1, 15)
        if max_c < 0:
            break
        new_color = random.randint(0, max_c)
        
        if new_color == old_color:
            continue
        
        # Check if recoloring maintains validity
        valid = True
        for k4_idx in edge_to_k4s[e_idx]:
            # Count distinct colors before and after
            e_list = k4_edges[k4_idx]
            old_seen = set()
            new_seen = set()
            for e2 in e_list:
                if e2 == e_idx:
                    old_seen.add(old_color)
                    new_seen.add(new_color)
                else:
                    old_seen.add(current[e2])
                    new_seen.add(current[e2])
            if len(new_seen) < 5:
                valid = False
                break
        
        if not valid:
            continue
        
        # Apply change
        current[e_idx] = new_color
        
        # Update K4 distinct counts
        for k4_idx in edge_to_k4s[e_idx]:
            k4_distinct[k4_idx] = count_distinct_colors_in_k4(k4_idx, current)
        
        new_max = get_max_color_used(current)
        
        # Objective: minimize max color
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max_color:
                best_max_color = new_max
                best_colors = current.copy()
        elif new_max == current_max:
            # Accept with probability
            prob = np.exp(-0.1 / temp)
            if random.random() < prob:
                pass  # Accept
            else:
                # Revert
                current[e_idx] = old_color
                for k4_idx in edge_to_k4s[e_idx]:
                    k4_distinct[k4_idx] = count_distinct_colors_in_k4(k4_idx, current)
        
        temp *= cool_rate
        if temp < 0.01:
            temp = 0.01
    
    # Final check
    if not check_validity(best_colors):
        # Fallback to original
        for idx, (i, j) in enumerate(edges):
            best_colors[idx] = (i + j) % 17
    
    return [int(c) for c in best_colors]
