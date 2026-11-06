import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def check_valid(adj, n, min_deg):
        # adj is a list of bitmasks
        # Check min degree
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        # Check C4
        # A C4 exists if there are u, v, w, x such that u-v, v-w, w-x, x-u are edges
        # Equivalently, any two vertices share at most 1 common neighbor.
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                # If common neighbors > 1, C4 exists
                if bin(common).count('1') > 1:
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            # Extract neighbors
            neighbors = bin(adj[i]).count('1')
            # Iterate through bits
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def solve_for_n(n, min_deg):
        # Use bitmasks for adjacency for speed
        # adj[i] is an integer where bit j is set if i-j is an edge
        
        # Strategy 1: Start with a random graph and repair
        # Strategy 2: Greedy construction
        
        best_adj = None
        best_score = -1
        
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 7919 + 13)
            
            # Initialize empty graph
            adj = [0] * n
            
            # Try to build a graph with high minimum degree
            # Greedy approach: randomly add edges if they don't create C4
            # This might be slow if we check every pair.
            # Better: Start with a dense random graph and remove C4s.
            
            # Density p = min_deg / n * 1.1
            p = min_deg / n * 1.1
            if p > 0.8: p = 0.8
            
            # Random initialization
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Repair Phase
            # 1. Remove C4s
            # 2. Add edges to satisfy min degree
            
            improved = True
            max_iter = 500
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Check for C4s
                c4_found = False
                # Sample some pairs to find C4s quickly, or check all?
                # For n=47, n^2/2 is ~1100 pairs. Checking all is fast with bitmasks.
                violating_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if bin(common).count('1') > 1:
                            violating_pairs.append((i, j))
                            c4_found = True
                            
                if c4_found:
                    # Pick a random violating pair
                    i, j = random.choice(violating_pairs)
                    common = adj[i] & adj[j]
                    # Get list of common neighbors
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    # Remove edges to break C4
                    # We need to reduce common neighbors to 1.
                    # Remove (i, w) or (j, w) for w in common_list[1:]
                    # Prefer removing from vertex with higher degree to preserve min_deg potential
                    deg_i = bin(adj[i]).count('1')
                    deg_j = bin(adj[j]).count('1')
                    
                    for w in common_list[1:]:
                        # Decide which edge to remove
                        if deg_i > deg_j:
                            # Remove (i, w)
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg_i -= 1
                        elif deg_j > deg_i:
                            # Remove (j, w)
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg_j -= 1
                        else:
                            # Random
                            if random.random() < 0.5:
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg_i -= 1
                            else:
                                adj[j] &= ~(1 << w)
                                adj[w] &= ~(1 << j)
                                deg_j -= 1
                else:
                    # No C4s, check min degree
                    min_deg_actual = min(bin(adj[i]).count('1') for i in range(n))
                    if min_deg_actual >= min_deg:
                        # Success!
                        if check_valid(adj, n, min_deg): # Double check
                            return adj
                        # If check_valid fails (shouldn't if logic above is correct), break
                        break
                    else:
                        # Need to add edges
                        # Find vertex with low degree
                        low_verts = [i for i in range(n) if bin(adj[i]).count('1') < min_deg]
                        if not low_verts:
                            break
                        u = random.choice(low_verts)
                        # Find non-neighbors of u that can be connected without creating C4
                        candidates = []
                        for v in range(n):
                            if v == u: continue
                            if not (adj[u] & (1 << v)):
                                # Check if adding (u,v) creates C4
                                # It creates C4 if common_neighbors(u, v) >= 1?
                                # No, it creates C4 if there is w such that u-w and v-w exist.
                                # So we need common_neighbors(u, v) == 0.
                                common = adj[u] & adj[v]
                                if common == 0:
                                    candidates.append(v)
                        
                        if candidates:
                            v = random.choice(candidates)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            # Cannot add any edge to u without creating C4
                            # Try another low vertex or break
                            pass
                            
        return None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        result_adj = solve_for_n(n, min_deg)
        
        if result_adj is not None:
            # Final verification
            if check_valid(result_adj, n, min_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}
                
    # Fallback: Return a valid small graph to ensure non-zero score
    # n=39, min_deg=0 is trivially valid (empty graph)
    # But we want to maximize n. Let's return n=39 with empty graph.
    return {'n': 39, 'edges': []}
