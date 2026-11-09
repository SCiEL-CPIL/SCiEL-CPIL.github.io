import time
import random
import itertools

def construct():
    n = 11
    vertices = list(range(n))
    all_edges = list(itertools.combinations(vertices, 5))
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Precompute masks for each edge
    # mask[i] is a bitset where bit j is 1 if coloring j makes edge i monochromatic
    masks = [0] * num_edges
    
    for i, e in enumerate(all_edges):
        # Calculate the bitmask for the vertices in this edge
        edge_bitmask = 0
        for v in e:
            edge_bitmask |= (1 << v)
        
        m = 0
        for c in range(num_colorings):
            val = c & edge_bitmask
            if val == 0 or val == edge_bitmask:
                m |= (1 << c)
        masks[i] = m

    best_len = float('inf')
    best_edges = []
    
    # Deadline guard: 100 seconds to be safe within 120s limit
    deadline = time.time() + 100.0
    
    while time.time() < deadline:
        active = (1 << num_colorings) - 1
        selected_indices = []
        
        while active != 0:
            # Find edges that cover at least one active coloring
            candidates = []
            for i in range(num_edges):
                # If the edge mask has no overlap with active, skip
                if masks[i] & active:
                    cnt = (masks[i] & active).bit_count()
                    candidates.append((cnt, i))
            
            if not candidates:
                # This should not happen for n=11, but break to avoid infinite loop
                break
            
            # Sort by count descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick randomly from top 10 to diversify search
            k = min(10, len(candidates))
            _, idx = random.choice(candidates[:k])
            
            selected_indices.append(idx)
            active &= ~masks[idx]
            
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_edges = [all_edges[i] for i in selected_indices]
            
            # If we found a very good solution, we can stop or continue to see if we can do better
            if best_len <= 51:
                break

    return {'n': n, 'edges': [list(e) for e in best_edges]}
