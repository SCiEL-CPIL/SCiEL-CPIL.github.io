import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 16
    # Generate all edges in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    num_edges = len(edges) # 120
    
    # Create edge index lookup
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        
    # Generate all K4s and their edge indices
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e_list = [
                        edge_index[(a, b)],
                        edge_index[(a, c)],
                        edge_index[(a, d)],
                        edge_index[(b, c)],
                        edge_index[(b, d)],
                        edge_index[(c, d)]
                    ]
                    k4s.append(e_list)
                    
    num_k4s = len(k4s) # 1820
    
    # For each edge, precompute which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def count_violations(colors):
        """Count number of K4s with < 5 distinct colors."""
        violations = 0
        for e_list in k4s:
            # Optimize: check if len(set(colors[i] for i in e_list)) < 5
            # Using a small set is fast enough
            if len(set(colors[i] for i in e_list)) < 5:
                violations += 1
        return violations
        
    def check_validity(colors):
        for e_list in k4s:
            if len(set(colors[i] for i in e_list)) < 5:
                return False
        return True
        
    # Fallback solution: (i+j) % 17
    seed_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    best_colors = seed_colors.copy()
    best_max_color = 17
    
    # Try to find solutions with fewer colors
    # We search for K=13, 14, 15
    # If we find a valid solution with K, we update best and continue to see if we can go lower?
    # No, if we find K=13, that's optimal (lower bound). 
    # If we find K=14, we can stop if we are confident, but let's try to improve.
    # Actually, finding K=13 is very hard. K=14 is hard. K=15 is easier.
    # Let's iterate K from 13 to 15.
    
    for target_k in [13, 14, 15]:
        if time.time() > deadline - 5.0:
            break
            
        found_valid = False
        # Multiple restarts
        max_restarts = 200
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Random initialization
            colors = np.random.randint(0, target_k, size=num_edges, dtype=np.int32)
            
            # Calculate initial violations
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                # Found valid solution
                if target_k < best_max_color:
                    best_colors = colors.copy()
                    best_max_color = target_k
                found_valid = True
                break # Try next K (lower) or exit loop? 
                # If we found 13, we can return immediately as it's the lower bound.
                # If we found 14, we might try 13? No, we are iterating 13, 14, 15.
                # So if we found 14, we already tried 13 and failed. So 14 is best so far.
                # We can break out of restart loop for this K.
                
            # Local Search
            # We want to minimize violations.
            # If we get to 0 violations, we have a valid solution.
            # Then we can try to reduce colors used? 
            # The problem asks to "Minimize the number of colours used".
            # If we fix the palette to [0, K-1], and we find a valid coloring, 
            # the number of colors used is <= K.
            # It might use fewer than K colors (e.g. if color 0 is never used).
            # We should check the actual max color used + 1.
            
            max_iters = 2000
            no_improve_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 1.0:
                    break
                    
                if current_violations == 0:
                    # We have a valid solution. 
                    # Check actual colors used
                    actual_max = int(np.max(colors)) + 1
                    if actual_max < best_max_color:
                        best_colors = colors.copy()
                        best_max_color = actual_max
                    found_valid = True
                    break
                    
                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                
                # Pick a new color
                new_color = random.randint(0, target_k - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta violations for K4s containing e_idx
                delta_v = 0
                affected_k4s = edge_to_k4s[e_idx]
                
                # Pre-fetch colors for faster access
                c_arr = colors
                
                for k4_idx in affected_k4s:
                    e_list = k4s[k4_idx]
                    
                    # Calculate distinct colors before
                    # We can optimize this by keeping a count, but for 6 items, set is fine
                    # However, doing this 91 times per move might be slow in Python.
                    # Let's try a slightly optimized version.
                    
                    # Get the other 5 edges
                    # e_list is a list of 6 indices. One is e_idx.
                    # We can just iterate e_list.
                    
                    # Old state
                    old_colors_set = set()
                    for e2 in e_list:
                        if e2 == e_idx:
                            old_colors_set.add(old_color)
                        else:
                            old_colors_set.add(c_arr[e2])
                            
                    old_distinct = len(old_colors_set)
                    
                    # New state
                    # We can update the set
                    # If old_color was unique in the set, removing it decreases size.
                    # But it's easier to just build the new set or check membership.
                    
                    # Let's just build the new set, it's only 6 items.
                    new_colors_set = set()
                    for e2 in e_list:
                        if e2 == e_idx:
                            new_colors_set.add(new_color)
                        else:
                            new_colors_set.add(c_arr[e2])
                            
                    new_distinct = len(new_colors_set)
                    
                    old_viol = 1 if old_distinct < 5 else 0
                    new_viol = 1 if new_distinct < 5 else 0
                    
                    delta_v += new_viol - old_viol
                    
                # Acceptance criteria
                if delta_v < 0:
                    # Improvement
                    colors[e_idx] = new_color
                    current_violations += delta_v
                    no_improve_count = 0
                elif delta_v == 0:
                    # Neutral move, accept with probability to escape local optima
                    if random.random() < 0.2:
                        colors[e_idx] = new_color
                        # current_violations unchanged
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    # Worsening, reject
                    no_improve_count += 1
                    
                if no_improve_count > 500:
                    break
                    
    # Final validity check
    if not check_validity(best_colors):
        # Should not happen, but fallback
        best_colors = seed_colors
        best_max_color = 17
        
    return [int(c) for c in best_colors]
