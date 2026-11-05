import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for each K4
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)
        
    k4_edges = []
    k4_vertices = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(edges)
                    k4_vertices.append((a,b,c,d))
                    
    num_k4s = len(k4_edges)
    
    # 1. Construct 1-factorization (15 colors)
    # Vertices 0..15. Vertex 15 is 'infinity'.
    # Matching k (0..14) contains edge (k, 15) and 7 other edges from 0..14 \ {k}
    # Pairs are ((k+i)%15, (k-i)%15) for i=1..7
    
    base_colors = [-1] * num_edges
    # Map edge to matching index
    for k in range(15):
        # Edge (k, 15)
        e = edge_idx(k, 15)
        base_colors[e] = k
        
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u == v: continue # Should not happen for i=1..7, mod 15
            # Ensure u, v are not k?
            # (k+i)%15 == k => i=0. (k-i)%15 == k => i=0.
            # So u,v != k.
            # Also u,v != 15 because mod 15.
            e = edge_idx(u, v)
            # Check if already assigned? No, 1-factorization is unique.
            if base_colors[e] == -1:
                base_colors[e] = k
            else:
                # This should not happen if construction is correct
                pass
                
    # Verify base_colors is a valid 15-coloring
    # Every K4 should have 6 distinct base colors
    # Since edges in K4 are distinct, and matchings are disjoint sets of edges,
    # the 6 edges belong to 6 distinct matchings.
    # So distinct colors = 6.
    
    # 2. Try to reduce colors by merging matchings
    # Current colors: 0..14
    # We want to map 0..14 to 0..k-1 to minimize k.
    # Constraint: For every K4, the 6 base colors must map to >= 5 distinct values.
    
    # Precompute the set of base colors for each K4
    k4_base_colors = []
    for edges in k4_edges:
        cols = set(base_colors[e] for e in edges)
        k4_base_colors.append(cols)
        
    # We use a greedy/local search to assign color groups.
    # Start with 15 groups (identity).
    # Try to merge groups.
    
    def check_validity(color_map):
        # color_map: list of length 15, color_map[i] is the new color for matching i
        # Actually, we just need to check if the number of distinct colors in each K4 is >= 5
        for cols in k4_base_colors:
            new_cols = set(color_map[c] for c in cols)
            if len(new_cols) < 5:
                return False
        return True
        
    def count_colors(color_map):
        return len(set(color_map))
    
    # Local search to minimize number of colors
    # State: assignment of 15 matchings to color labels
    # We can use a simple greedy: 
    # Start with identity.
    # Try to merge color A into color B.
    # If valid, do it.
    # Repeat until no more merges possible.
    # Then try random restarts or swaps to find better merges.
    
    best_assignment = list(range(15))
    best_k = 15
    
    # Optimization: Precompute for each pair of matchings (a, b), 
    # if merging a and b causes any K4 to drop below 5 colors?
    # This depends on the current assignment.
    # It's easier to just check validity after each merge attempt.
    # 1820 K4s * 15 colors check is fast.
    
    def try_merge(assign, a, b):
        # Try to merge color a into color b in assignment 'assign'
        # Return new assignment if valid, else None
        new_assign = assign[:]
        for i in range(15):
            if new_assign[i] == a:
                new_assign[i] = b
        # Check validity
        for cols in k4_base_colors:
            new_cols = set(new_assign[c] for c in cols)
            if len(new_cols) < 5:
                return None
        return new_assign
        
    # Greedy merge
    current_assign = list(range(15))
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Get unique colors currently used
        used_colors = set(current_assign)
        # Try to merge each pair
        # To be efficient, iterate over pairs of colors present
        color_list = list(used_colors)
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1, c2 = color_list[i], color_list[j]
                new_assign = try_merge(current_assign, c1, c2)
                if new_assign is not None:
                    current_assign = new_assign
                    improved = True
                    break
            if improved:
                break
                
    if count_colors(current_assign) < best_k:
        best_k = count_colors(current_assign)
        best_assignment = current_assign[:]
        
    # Local search / Random restarts to find better merges
    # We can randomize the initial assignment or the merge order.
    
    for _ in range(100):
        if time.time() > deadline - 10:
            break
            
        # Randomize initial assignment slightly? 
        # Or just restart greedy with random order of merge attempts.
        current_assign = list(range(15))
        
        # Shuffle the order of colors to try merging
        colors_to_try = list(range(15))
        random.shuffle(colors_to_try)
        
        # Greedy pass with shuffled order
        # We need to be careful: merging changes the set of colors.
        # Simple approach: repeatedly pick a random pair of colors present, try to merge.
        steps = 0
        while steps < 100: # Max steps per restart
            steps += 1
            used = set(current_assign)
            if len(used) <= best_k:
                break
            color_list = list(used)
            if len(color_list) < 2:
                break
            random.shuffle(color_list)
            merged = False
            for i in range(len(color_list)):
                for j in range(i+1, len(color_list)):
                    c1, c2 = color_list[i], color_list[j]
                    new_assign = try_merge(current_assign, c1, c2)
                    if new_assign is not None:
                        current_assign = new_assign
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
                
        k_curr = count_colors(current_assign)
        if k_curr < best_k:
            best_k = k_curr
            best_assignment = current_assign[:]
            
    # Final Polish: Try to merge any remaining pairs in best_assignment
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        used = set(best_assignment)
        color_list = list(used)
        random.shuffle(color_list)
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1, c2 = color_list[i], color_list[j]
                new_assign = try_merge(best_assignment, c1, c2)
                if new_assign is not None:
                    best_assignment = new_assign
                    improved = True
                    break
            if improved:
                break
                
    # Map base colors to final colors
    # Remap 0..best_k-1
    final_colors = [0] * num_edges
    # Create mapping from old color label to new 0..k-1
    # The values in best_assignment are the labels.
    # We just need to ensure the output uses 0..k-1
    # The set of values in best_assignment has size best_k.
    # We can just use the values directly if they are compacted, or remap.
    # try_merge keeps values as they are (merges a into b, so a disappears).
    # So the values are a subset of 0..14.
    # The evaluator counts distinct integers in the output.
    # So we can just return the values in best_assignment.
    
    for e in range(num_edges):
        final_colors[e] = best_assignment[base_colors[e]]
        
    return final_colors
