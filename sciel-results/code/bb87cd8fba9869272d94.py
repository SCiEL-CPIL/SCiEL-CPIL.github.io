"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    best_result = {'n': 39, 'edges': []}
    
    for target_n in [47, 46, 45]:
        if elapsed() > deadline:
            break
        min_deg_req = target_n - 39
        
        for seed in range(30):
            if elapsed() > deadline:
                break
            
            random.seed(seed * 7919 + target_n * 131)
            
            # Bitmask representation: adj[v] is an int with bit u set iff (u,v) in E
            adj = [0] * target_n
            
            def add_edge(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            def remove_edge(u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            
            def has_c4_if_add(u, v):
                # Check if adding (u,v) creates a C4: need |N(u) & N(v)| >= 2
                common = adj[u] & adj[v]
                return common.bit_count() >= 2
            
            def min_degree():
                return min(adj[v].bit_count() for v in range(target_n))
            
            # Phase 1: Build using polarity graph structure of PG(2, q) as seed
            # For q=7: PG(2,7) has 57 points. Use a subset or full.
            # Actually, let's use a different structured seed: 
            # Start with a random graph biased toward high degree, then repair C4s.
            
            # Strategy: Start dense, then remove edges to break C4s while maintaining min degree
            # Initialize with all edges
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    add_edge(u, v)
            
            # Phase 1: Remove edges to make C4-free, trying to preserve degrees
            # Greedy: find C4s and remove one edge from each
            max_removals = 5000
            for _ in range(max_removals):
                if elapsed() > deadline:
                    break
                # Find a C4: look for two vertices with >= 2 common neighbors
                found_c4 = False
                for u in range(target_n):
                    if elapsed() > deadline:
                        break
                    nu = adj[u]
                    # For each pair of neighbors of u, check if they share another common neighbor
                    # Faster: for each v not adjacent to u, check |N(u) & N(v)| >= 2
                    for v in range(target_n):
                        if v == u or (nu >> v) & 1:
                            continue
                        common = nu & adj[v]
                        if common.bit_count() >= 2:
                            # Found C4: u, a, v, b where a,b in common
                            # Remove edge (a, b) or (u, a) or (v, b) etc.
                            # Remove the edge that minimally hurts degrees
                            # Try removing (a,b) for a random common neighbor pair
                            common_list = [x for x in range(target_n) if (common >> x) & 1]
                            if len(common_list) >= 2:
                                a = random.choice(common_list)
                                b = random.choice([x for x in common_list if x != a])
                                # Remove edge (a, b)
                                if (adj[a] >> b) & 1:
                                    remove_edge(a, b)
                                    found_c4 = True
                                    break
                    if found_c4:
                        break
                if not found_c4:
                    break
            
            # Phase 2: Local search with swaps to increase min degree
            for iteration in range(300):
                if elapsed() > deadline:
                    break
                
                degs = [adj[v].bit_count() for v in range(target_n)]
                if min(degs) >= min_deg_req:
                    break
                
                # Find low-degree vertex
                low_v = min(range(target_n), key=lambda v: degs[v])
                if degs[low_v] >= min_deg_req:
                    break
                
                # Try to add an edge from low_v to a non-neighbor without creating C4
                non_neighbors = [w for w in range(target_n) if w != low_v and not ((adj[low_v] >> w) & 1)]
                random.shuffle(non_neighbors)
                
                improved = False
                for w in non_neighbors:
                    if has_c4_if_add(low_v, w):
                        continue
                    add_edge(low_v, w)
                    degs[low_v] += 1
                    degs[w] += 1
                    improved = True
                    break
                
                if not improved:
                    # Try a swap: remove edge (a,b) and add (low_v, w)
                    # Find an edge whose removal doesn't drop any degree below min_deg_req
                    # and whose removal allows adding (low_v, w)
                    for a in range(target_n):
                        if degs[a] <= min_deg_req:
                            continue
                        for b in range(a + 1, target_n):
                            if not ((adj[a] >> b) & 1):
                                continue
                            if degs[b] <= min_deg_req:
                                continue
                            # Try removing (a,b)
                            remove_edge(a, b)
                            degs[a] -= 1
                            degs[b] -= 1
                            
                            # Now try to add (low_v, w) for some w
                            added = False
                            non_neighbors2 = [w for w in range(target_n) if w != low_v and not ((adj[low_v] >> w) & 1)]
                            random.shuffle(non_neighbors2)
                            for w in non_neighbors2:
                                if has_c4_if_add(low_v, w):
                                    continue
                                add_edge(low_v, w)
                                degs[low_v] += 1
                                degs[w] += 1
                                added = True
                                break
                            
                            if not added:
                                # Revert
                                add_edge(a, b)
                                degs[a] += 1
                                degs[b] += 1
                            
                            break
                    
            # Verify
            degs = [adj[v].bit_count() for v in range(target_n)]
            if min(degs) >= min_deg_req:
                # Full C4 verification
                is_c4_free = True
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if not ((adj[u] >> v) & 1):  # non-adjacent
                            common = adj[u] & adj[v]
                            if common.bit_count() >= 2:
                                is_c4_free = False
                                break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    edges = []
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    best_result = {'n': target_n, 'edges': edges}
                    if target_n >= 46:
                        return best_result
                    break
    
    return best_result
# EVOLVE-BLOCK-END
