import random
import time
import sys
from itertools import combinations

def construct():
    n = 16
    num_edges = 120
    
    # Map (i, j) to index
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    # Precompute K4s
    k4s = []
    k4_edge_indices = []
    
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
        return len(set(colors_6)) >= 5

    def min_conflicts_search(K, time_limit):
        # Initialize random colors
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # Calculate initial conflict counts
        k4_bad = [0] * num_k4s
        edge_bad = [0] * num_edges
        
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            c = [colors[e] for e in e_indices]
            if not is_valid_k4(c):
                k4_bad[k4_idx] = 1
                for e in e_indices:
                    edge_bad[e] += 1
        
        total_bad = sum(k4_bad)
        
        if total_bad == 0:
            return colors
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random edge that is in a bad K4
            # To be efficient, we can maintain a list of bad edges, but 
            # simple random selection with rejection is often fast enough
            # given the high density of constraints.
            
            # A better way: pick a random bad K4, then pick a random edge in it.
            # This ensures we pick an edge that is definitely in a conflict.
            if total_bad == 0:
                return colors
                
            # Pick a random bad K4
            # We can maintain a list of bad k4 indices for O(1) access
            # But for simplicity and speed, let's just pick a random k4 and check?
            # No, that's inefficient if most are good.
            
            # Let's maintain a list of bad k4s.
            # Actually, let's just pick a random edge and check if it's bad.
            # If not, try again.
            # Or, pick a random k4, if it's bad, pick a random edge in it.
            
            # We need a list of bad k4s to pick from efficiently.
            # Updating this list dynamically is complex.
            # Let's use the edge_bad array.
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            if edge_bad[e_idx] == 0:
                continue
            
            # Try all colors for this edge
            best_color = colors[e_idx]
            min_bad_increase = float('inf')
            
            # We calculate the change in total bad K4s if we change e_idx to c
            # Only K4s containing e_idx are affected
            affected_k4s = edge_to_k4s[e_idx]
            
            # We need to evaluate the cost for each candidate color
            # Cost = number of bad K4s among affected_k4s
            
            # Optimization: 
            # For each affected K4, we can check if it becomes bad/good.
            
            for c in range(K):
                if c == colors[e_idx]:
                    continue
                
                current_bad = 0
                for k4_idx in affected_k4s:
                    e_indices = k4_edge_indices[k4_idx]
                    # Construct color list for this K4 with new color for e_idx
                    # We can do this efficiently
                    # Instead of copying the whole list, we can check the set size
                    # But we need to know if the set size >= 5
                    
                    # Let's get the colors of the other 5 edges
                    # And add the new color c
                    
                    # To be fast, we can pre-store the colors of the other edges?
                    # No, they change.
                    
                    # Simple check:
                    # Get the 6 colors, replace e_idx with c, check set size.
                    # This is 6 operations per K4.
                    # 91 K4s * 6 ops = 546 ops per color.
                    # K colors (e.g., 13) -> 13 * 546 ~ 7000 ops per move.
                    # This is acceptable.
                    
                    c_list = []
                    for e in e_indices:
                        if e == e_idx:
                            c_list.append(c)
                        else:
                            c_list.append(colors[e])
                            
                    if len(set(c_list)) < 5:
                        current_bad += 1
                        
                if current_bad < min_bad_increase:
                    min_bad_increase = current_bad
                    best_color = c
                    
            # Update colors
            old_color = colors[e_idx]
            if best_color != old_color:
                colors[e_idx] = best_color
                
                # Update conflict counts
                # We need to recalculate k4_bad and edge_bad for affected K4s
                # This is expensive if we do it naively.
                
                # Let's just recalculate total_bad at the end of the loop?
                # No, we need edge_bad to pick the next edge.
                
                # We can update incrementally.
                # For each affected K4:
                #   Check if it was bad (k4_bad[k4_idx])
                #   Check if it is bad now
                #   If status changed, update k4_bad and edge_bad for all 6 edges in that K4.
                
                for k4_idx in affected_k4s:
                    e_indices = k4_edge_indices[k4_idx]
                    
                    # Check if bad now
                    c_list = [colors[e] for e in e_indices]
                    is_bad_now = 0 if len(set(c_list)) >= 5 else 1
                    
                    was_bad = k4_bad[k4_idx]
                    
                    if is_bad_now != was_bad:
                        k4_bad[k4_idx] = is_bad_now
                        total_bad += (is_bad_now - was_bad)
                        
                        # Update edge_bad for all edges in this K4
                        for e in e_indices:
                            edge_bad[e] += (is_bad_now - was_bad)
                            
            if total_bad == 0:
                return colors
                
        return None

    # Try K=13, 14, 15
    # Allocate time
    # 13 is hardest. Give it more time? Or try it first with a short time, then 14?
    # If 13 is impossible, we waste time.
    # But we want to minimize.
    # Let's try 13 for 20s, 14 for 20s, 15 for 20s.
    # If any succeeds, return.
    
    # Fallback
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    result = None
    for K in [13, 14, 15]:
        # Run multiple restarts
        start = time.time()
        time_left = 20
        while time.time() - start < time_left:
            res = min_conflicts_search(K, time_limit=5) # 5s per run
            if res is not None:
                result = res
                break
        if result:
            break
            
    if result:
        return result
    else:
        return fallback
