import random
import time
import sys

def construct():
    n = 16
    num_edges = 120
    # Map (i, j) to index
    # Index = i * (n - i - 1) // 2 + (j - i - 1) is not quite right for simple loop
    # Let's just generate list of edges and map
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    # Precompute K4s
    # A K4 is a set of 4 vertices {v0, v1, v2, v3}
    # It contains 6 edges: (v0,v1), (v0,v2), (v0,v3), (v1,v2), (v1,v3), (v2,v3)
    k4s = []
    k4_edge_indices = []
    
    # Generate all combinations of 4 vertices
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            for v2 in range(v1 + 1, n):
                for v3 in range(v2 + 1, n):
                    verts = [v0, v1, v2, v3]
                    e_indices = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            e = (verts[a], verts[b])
                            e_indices.append(edge_to_idx[e])
                    k4s.append(verts)
                    k4_edge_indices.append(e_indices)
                    
    num_k4s = len(k4s)
    
    # Map edge index to list of k4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4_edge_indices[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)

    def is_valid_k4(colors_6):
        # Check if at least 5 distinct colors
        return len(set(colors_6)) >= 5

    def count_bad_k4s(colors):
        bad = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            c = [colors[e] for e in e_indices]
            if not is_valid_k4(c):
                bad += 1
        return bad

    def min_conflicts_search(K, time_limit):
        # Initialize random colors
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # Initial bad K4 set
        bad_set = set()
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            c = [colors[e] for e in e_indices]
            if not is_valid_k4(c):
                bad_set.add(k4_idx)
                
        if not bad_set:
            return colors

        start_time = time.time()
        
        while bad_set and time.time() - start_time < time_limit:
            # Pick a random bad K4
            bad_idx = random.choice(list(bad_set))
            e_indices = k4_edge_indices[bad_idx]
            
            # Pick a random edge from this K4
            target_edge = random.choice(e_indices)
            old_color = colors[target_edge]
            
            # Identify affected K4s
            affected_k4s = edge_to_k4s[target_edge]
            
            # Pre-compute "other colors" sets for affected K4s
            # To speed up evaluation of new colors
            k4_data = []
            for k4_idx in affected_k4s:
                e_idx_list = k4_edge_indices[k4_idx]
                other_colors = []
                for e in e_idx_list:
                    if e != target_edge:
                        other_colors.append(colors[e])
                k4_data.append((k4_idx, set(other_colors)))
                
            # Evaluate all K colors
            best_delta = float('inf')
            best_colors = []
            
            for c in range(K):
                if c == old_color:
                    continue
                    
                delta = 0
                for k4_idx, S_other in k4_data:
                    # Determine if K4 is currently valid
                    # Current color is old_color
                    if old_color in S_other:
                        size_old = len(S_other)
                    else:
                        size_old = len(S_other) + 1
                        
                    # Determine if K4 is valid with new color c
                    if c in S_other:
                        size_new = len(S_other)
                    else:
                        size_new = len(S_other) + 1
                        
                    old_valid = size_old >= 5
                    new_valid = size_new >= 5
                    
                    if old_valid and not new_valid:
                        delta += 1
                    elif not old_valid and new_valid:
                        delta -= 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_colors = [c]
                elif delta == best_delta:
                    best_colors.append(c)
                    
            # Pick a random best color
            new_color = random.choice(best_colors)
            
            if new_color != old_color:
                colors[target_edge] = new_color
                
                # Update bad_set
                # Re-evaluate affected K4s
                # We need to remove those that are now valid, add those that are now invalid.
                
                # To do this efficiently, we check each affected K4.
                # Note: A K4 might be in bad_set or not.
                
                for k4_idx, S_other in k4_data:
                    # Check current validity with new_color
                    if new_color in S_other:
                        size_new = len(S_other)
                    else:
                        size_new = len(S_other) + 1
                        
                    is_valid_now = size_new >= 5
                    
                    if is_valid_now:
                        # If it was bad, remove it
                        if k4_idx in bad_set:
                            bad_set.remove(k4_idx)
                    else:
                        # If it was good, add it
                        if k4_idx not in bad_set:
                            bad_set.add(k4_idx)
                            
        return colors if not bad_set else None

    # Strategy:
    # Try K=13, then 14, then 15.
    # Allocate time.
    # If K=13 succeeds, return.
    
    # Fallback solution (17 colors) - though 15 should be easy
    # A simple modular coloring might work for 15?
    # (i+j) % 15?
    # Let's just rely on the search.
    
    result = None
    # Time allocation:
    # 13: 30s
    # 14: 40s
    # 15: 40s
    # Total 110s, leaving 10s for overhead.
    
    for K in [13, 14, 15]:
        t_limit = 30 if K == 13 else 40
        start = time.time()
        while time.time() - start < t_limit:
            res = min_conflicts_search(K, time_limit=5) # 5s per run, multiple restarts
            if res is not None:
                # Verify
                if count_bad_k4s(res) == 0:
                    result = res
                    break
        if result:
            break
            
    if result:
        return result
    else:
        # Fallback: 15 colors
        # Try one more time with K=15 if not found?
        # Or just return a known valid 15-coloring if we had one.
        # Since 15 is very likely to be found, this is unlikely.
        # Let's do a quick K=16 fallback just in case.
        res = min_conflicts_search(16, time_limit=5)
        if res is not None:
            return res
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
