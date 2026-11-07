import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Identify tree edges and non-tree edges
    # Tree:
    # Top path: (0,1), (1,2), ..., (m-2, m-1)
    # Bottom path: (m, m+1), ..., (2m-2, 2m-1)
    # Matching: (0, m)
    
    tree_edges = set()
    for i in range(m-1):
        tree_edges.add((i, i+1))
        tree_edges.add((m+i, m+i+1))
    tree_edges.add((0, m))
    
    # Note: Edges in graph are (u, v). We need to map them to indices.
    # Since the graph is undirected, (u,v) and (v,u) are the same.
    # We should normalize the edge representation for lookup.
    edge_index_map = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_index_map[(u, v)] = i
        
    tree_edge_indices = []
    non_tree_edge_indices = []
    
    for i, (u, v) in enumerate(edges):
        u_norm, v_norm = (u, v) if u < v else (v, u)
        if (u_norm, v_norm) in tree_edges:
            tree_edge_indices.append(i)
        else:
            non_tree_edge_indices.append(i)
            
    k = len(non_tree_edge_indices)
    
    best_signs = None
    best_radius = float('inf')
    
    # Deadline guard
    start_time = time.time()
    timeout = 50 # Leave 10s margin
    
    # Enumerate all 2^k sign assignments for non-tree edges
    # Tree edges are fixed to +1
    for mask in range(2**k):
        if time.time() - start_time > timeout:
            break
            
        signs = np.ones(n_edges, dtype=np.int8)
        
        # Set non-tree edges based on mask
        for j in range(k):
            if (mask >> j) & 1:
                signs[non_tree_edge_indices[j]] = -1
            else:
                signs[non_tree_edge_indices[j]] = 1
                
        # Tree edges are already +1
        
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            pass # Invalid lift
            
    if best_signs is None:
        # Fallback: all +1 might be invalid if not connected?
        # Prism graph lift with all +1 is two disjoint copies, so it's NOT connected.
        # So we must find a valid one.
        # If the exhaustive search fails (shouldn't happen), try random or a heuristic.
        # But for m<=9, there are definitely valid connected lifts.
        # E.g., flipping just one matching edge sign usually connects the two components?
        # Actually, if all signs are +1, the lift is two disjoint copies of the base graph.
        # To connect them, we need at least one -1 sign.
        # Let's just return the best found. If none found, raise or return a default?
        # The problem states "The return must have shape... and contain only -1 and +1".
        # And "If invalid... scores 0". So we must return a valid one.
        # Given the structure, there are many valid ones.
        
        # Fallback: try a specific pattern known to be connected.
        # e.g. all +1 except one matching edge is -1.
        signs_fallback = np.ones(n_edges, dtype=np.int8)
        # Find index of edge (0, m)
        idx_0_m = edge_index_map.get((0, m), edge_index_map.get((m, 0)))
        if idx_0_m is not None:
            signs_fallback[idx_0_m] = -1
        try:
             radius, _ = check_lift(graph, signs_fallback)
             return signs_fallback
        except:
             return np.ones(n_edges, dtype=np.int8) # Last resort, likely invalid but prevents crash

    return best_signs
