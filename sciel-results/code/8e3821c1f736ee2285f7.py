import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = [] 
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Function to check if a specific edge change is valid
    def can_change_edge(e_idx, new_color, colors):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            # Calculate colors of other 5 edges
            # We can optimize by just counting distincts
            # Since the set is small, a simple list comprehension and set is fast enough
            # But let's try to be slightly more efficient
            other_colors = []
            for e in e_list:
                if e != e_idx:
                    other_colors.append(colors[e])
            
            s = set(other_colors)
            l_s = len(s)
            if new_color in s:
                if l_s < 5:
                    return False
            else:
                if l_s < 4:
                    return False
        return True

    # 1. Generate 1-factorization of K16
    # Vertices 0..14, 15 is infinity
    # Factors 0..14
    factor_assignment = [0] * num_edges
    for k in range(15):
        # Pair infinity (15) with k
        u, v = 15, k
        if u > v: u, v = v, u
        factor_assignment[edge_index[(u, v)]] = k
        
        for j in range(1, 8): # 1 to 7
            u = (k + j) % 15
            v = (k - j) % 15
            if u > v: u, v = v, u
            factor_assignment[edge_index[(u, v)]] = k
            
    # Initial colors: 0..14
    colors = factor_assignment[:]
    best_colors = colors[:]
    best_max_color = 14
    
    # Local Search
    # We want to minimize the number of distinct colors.
    # We do this by trying to map colors to lower indices.
    
    def compact(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    # Main loop
    while time.time() < deadline:
        improved = False
        # Try a batch of random edges
        # To speed up convergence, we can try to find the minimum valid color for each edge
        # But random sampling is often sufficient and faster per step.
        
        # Heuristic: Try to lower colors.
        # Pick a random edge. Try to set it to the smallest valid color.
        
        # To avoid getting stuck in a local minimum where no single edge can be lowered,
        # we occasionally perform a "kick": change an edge to a random valid color (possibly higher).
        
        for _ in range(2000):
            if time.time() > deadline:
                break
                
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            # Try to find the minimum valid color
            min_valid = -1
            # We can check 0, 1, ... up to current_c - 1
            # But checking all is expensive.
            # Let's check a few random lower colors, and 0.
            candidates = [0]
            if current_c > 1:
                candidates.append(random.randint(0, current_c - 1))
            
            for c in candidates:
                if c < current_c and can_change_edge(e_idx, c, colors):
                    min_valid = c
                    break
                    
            if min_valid != -1:
                colors[e_idx] = min_valid
                improved = True
            else:
                # If we couldn't lower it, maybe we can change it to a different valid color
                # to allow other edges to lower?
                # This is the kick.
                # Only kick if we are stuck for a while? 
                # For now, just try to lower.
                pass
        
        if improved:
            # Compact to see if we reduced the max color count
            # Compaction is O(E log E) roughly. 120 edges. Very fast.
            compacted = compact(colors)
            current_max = max(compacted)
            if current_max < best_max_color:
                best_max_color = current_max
                best_colors = compacted[:]
                colors = compacted[:]
                # After a significant improvement, we might want to restart the search phase
                # or just continue.
        else:
            # Stuck. Try a kick.
            # Pick a random edge and change it to a random valid color in [0, best_max_color]
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            found_kick = False
            for _ in range(5):
                new_c = random.randint(0, best_max_color)
                if new_c == current_c:
                    continue
                if can_change_edge(e_idx, new_c, colors):
                    colors[e_idx] = new_c
                    found_kick = True
                    break
            
            if not found_kick:
                # If we can't kick, maybe compact and try again?
                # Or just continue.
                pass

    # Final compact
    best_colors = compact(best_colors)
    
    return best_colors
