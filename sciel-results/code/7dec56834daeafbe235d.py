import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # 1. Initial Construction: K7 U K7 U K2
    # Vertices 0..6 -> K7
    # Vertices 7..13 -> K7
    # Vertices 14..15 -> K2
    edges = set()
    
    # Add K7 (0..6)
    for i in range(7):
        for j in range(i+1, 7):
            edges.add((i, j))
            
    # Add K7 (7..13)
    for i in range(7, 14):
        for j in range(i+1, 14):
            edges.add((i, j))
            
    # Add K2 (14, 15)
    edges.add((14, 15))
    
    # Precompute all 8-vertex subsets
    vertices = list(range(16))
    all_subsets = list(combinations(vertices, 8))
    
    # Map subset (tuple) to a list of edges within it? No, just check on the fly.
    # To speed up "does adding (u,v) create Q3", we need to check subsets containing u and v.
    # Precompute for each pair (u,v) the list of 8-subsets containing u and v.
    # There are 120 pairs. Each appears in C(14,6) = 3003 subsets.
    # Total storage: 120 * 3003 * 8 ints. ~3.6 million ints. ~28MB. Acceptable.
    subsets_by_pair = {}
    for subset in all_subsets:
        # Get all pairs in this subset
        for i in range(8):
            for j in range(i+1, 8):
                u = subset[i]
                v = subset[j]
                if u > v:
                    u, v = v, u
                if (u, v) not in subsets_by_pair:
                    subsets_by_pair[(u, v)] = []
                subsets_by_pair[(u, v)].append(subset)
                
    # Helper function to check if a set of edges on a specific 8-subset forms Q3
    def is_q3_subgraph(subset, edges_set):
        # Check edge count in induced subgraph
        e_count = 0
        # We need to check degrees and bipartiteness
        # Let's build the induced subgraph adjacency
        # Vertices are 0..15, but we only care about those in subset
        
        # Fast check 1: Edge count
        # Iterate through edges in edges_set? No, that's slow (up to 120 edges).
        # Better: iterate through pairs in subset.
        # Subset has 8 vertices, so 28 pairs.
        for i in range(8):
            for j in range(i+1, 8):
                u = subset[i]
                v = subset[j]
                if (u, v) in edges_set or (v, u) in edges_set:
                    e_count += 1
        if e_count != 12:
            return False
            
        # Fast check 2: Degrees
        degrees = {v: 0 for v in subset}
        for i in range(8):
            for j in range(i+1, 8):
                u = subset[i]
                v = subset[j]
                if (u, v) in edges_set or (v, u) in edges_set:
                    degrees[u] += 1
                    degrees[v] += 1
        if any(deg != 3 for deg in degrees.values()):
            return False
            
        # Fast check 3: Bipartite
        # BFS/DFS on the induced subgraph
        # Since it's 3-regular and 8 vertices, if it's bipartite, it's Q3.
        
        # Build adjacency list for subset
        adj = {v: [] for v in subset}
        for i in range(8):
            for j in range(i+1, 8):
                u = subset[i]
                v = subset[j]
                if (u, v) in edges_set or (v, u) in edges_set:
                    adj[u].append(v)
                    adj[v].append(u)
                    
        # Color vertices
        color = {}
        for v in subset:
            if v in color:
                continue
            color[v] = 0
            queue = [v]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[curr]
                        queue.append(neighbor)
                    else:
                        if color[neighbor] == color[curr]:
                            return False
        return True

    # Local Search
    # We maintain `edges` as a set of tuples (u,v) with u < v for O(1) lookup.
    # We will try to add edges.
    
    # List of all possible edges
    all_possible_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    # Shuffle to randomize order
    random.shuffle(all_possible_edges)
    
    current_edges = edges.copy()
    best_edges = current_edges.copy()
    best_score = len(best_edges)
    
    # We will iterate through non-edges and try to add them.
    # If we can't add one, we might try a swap (remove one, add one) if we were doing that.
    # For now, just greedy addition.
    
    # To allow restarts or different orders, we can just use the shuffled list.
    
    # Optimization: 
    # When checking if adding (u,v) is safe, we only need to check subsets in subsets_by_pair[(u,v)].
    
    for u, v in all_possible_edges:
        if time.time() > deadline:
            break
            
        if (u, v) in current_edges:
            continue
            
        # Check if adding (u, v) creates a Q3
        is_safe = True
        rel_subsets = subsets_by_pair.get((u, v), [])
        # Note: (u,v) might be stored as (min, max). Ensure u < v
        if u > v:
            u, v = v, u
        
        for subset in rel_subsets:
            # Check if G[subset] + {(u,v)} is Q3
            # We need to temporarily add (u,v) to the check?
            # Or just pass the set of edges + (u,v)
            # To avoid modifying `current_edges`, we can check:
            # Does `current_edges` contain a Q3 in `subset`? No, it's Q3-free.
            # Does `current_edges` U {(u,v)} contain a Q3 in `subset`?
            # This is equivalent to: is the induced subgraph of `current_edges` on `subset`, plus edge (u,v), isomorphic to Q3?
            
            # We can reuse is_q3_subgraph but we need to include (u,v).
            # Let's modify is_q3_subgraph to accept an extra edge.
            
            # Actually, is_q3_subgraph(subset, current_edges) is False (since current_edges is Q3-free).
            # We need to check the *new* state.
            
            # Let's create a helper that checks specific subset with specific edge set + 1 extra edge
            def is_q3_with_extra(subset, edges_set, extra_edge):
                u_e, v_e = extra_edge
                # Check edge count
                e_count = 0
                for i in range(8):
                    for j in range(i+1, 8):
                        uu = subset[i]
                        vv = subset[j]
                        if (uu, vv) in edges_set or (vv, uu) in edges_set:
                            e_count += 1
                if (u_e, v_e) in edges_set or (v_e, u_e) in edges_set:
                    pass # Already counted? No, we add it if not present.
                    # Wait, the extra edge is NOT in edges_set.
                e_count += 1 # Add the extra edge
                
                if e_count != 12:
                    return False
                
                # Check degrees
                degrees = {v: 0 for v in subset}
                for i in range(8):
                    for j in range(i+1, 8):
                        uu = subset[i]
                        vv = subset[j]
                        if (uu, vv) in edges_set or (vv, uu) in edges_set:
                            degrees[uu] += 1
                            degrees[vv] += 1
                
                # Add extra edge to degrees
                if u_e in degrees and v_e in degrees:
                    degrees[u_e] += 1
                    degrees[v_e] += 1
                else:
                    # Extra edge must be within the subset
                    return False
                
                if any(deg != 3 for deg in degrees.values()):
                    return False
                
                # Check Bipartite
                adj = {v: [] for v in subset}
                for i in range(8):
                    for j in range(i+1, 8):
                        uu = subset[i]
                        vv = subset[j]
                        if (uu, vv) in edges_set or (vv, uu) in edges_set:
                            adj[uu].append(vv)
                            adj[vv].append(uu)
                
                adj[u_e].append(v_e)
                adj[v_e].append(u_e)
                
                color = {}
                for v in subset:
                    if v in color:
                        continue
                    color[v] = 0
                    queue = [v]
                    while queue:
                        curr = queue.pop(0)
                        for neighbor in adj[curr]:
                            if neighbor not in color:
                                color[neighbor] = 1 - color[curr]
                                queue.append(neighbor)
                            else:
                                if color[neighbor] == color[curr]:
                                    return False
                return True

            if is_q3_with_extra(subset, current_edges, (u, v)):
                is_safe = False
                break
                
        if is_safe:
            current_edges.add((u, v))
            if len(current_edges) > best_score:
                best_score = len(current_edges)
                best_edges = current_edges.copy()
                
    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
