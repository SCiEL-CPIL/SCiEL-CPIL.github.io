# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _spectral_partition(vertices, adj, remaining_edges, max_depth=3):
    """Recursively partition vertices using spectral bisection."""
    if not vertices or len(vertices) <= 2 or max_depth <= 0:
        return [list(vertices)]
    
    n = len(vertices)
    v_list = sorted(vertices)
    
    # Build subgraph adjacency matrix
    A = np.zeros((n, n))
    for i, u in enumerate(v_list):
        for j, v in enumerate(v_list):
            if i != j and v in adj[u]:
                edge_key = (min(u, v), max(u, v))
                if edge_key in remaining_edges:
                    A[i, j] = 1
    
    # Compute eigenvectors of A
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    except:
        return [list(vertices)]
    
    # Use the second smallest eigenvector (Fiedler vector) for partitioning
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    fiedler = eigenvectors[:, idx[1]] if n > 1 else np.zeros(n)
    
    # Partition based on sign of Fiedler vector
    mean_val = np.mean(fiedler)
    cluster1 = [v_list[i] for i in range(n) if fiedler[i] >= mean_val]
    cluster2 = [v_list[i] for i in range(n) if fiedler[i] < mean_val]
    
    if not cluster1 or not cluster2:
        return [list(vertices)]
    
    # Recursively partition each cluster
    result = []
    result.extend(_spectral_partition(set(cluster1), adj, remaining_edges, max_depth - 1))
    result.extend(_spectral_partition(set(cluster2), adj, remaining_edges, max_depth - 1))
    
    return result

def _find_maximal_biclique_from_seed(seed_edge, adj, remaining_edges):
    """Find a maximal biclique starting from a seed edge."""
    u, v = seed_edge
    L = {u}
    R = {v}
    
    changed = True
    while changed:
        changed = False
        
        # Try to add vertices to L
        new_L_candidates = set()
        for r in R:
            new_L_candidates.update(adj[r])
        
        valid_new_L = set()
        for candidate in new_L_candidates:
            if candidate in R:
                continue
            is_connected = True
            for r in R:
                edge_key = (min(candidate, r), max(candidate, r))
                if edge_key not in remaining_edges:
                    is_connected = False
                    break
            if is_connected:
                valid_new_L.add(candidate)
        
        for vertex in valid_new_L:
            if vertex not in L:
                L.add(vertex)
                changed = True
        
        # Try to add vertices to R
        new_R_candidates = set()
        for l in L:
            new_R_candidates.update(adj[l])
        
        valid_new_R = set()
        for candidate in new_R_candidates:
            if candidate in L:
                continue
            is_connected = True
            for l in L:
                edge_key = (min(candidate, l), max(candidate, l))
                if edge_key not in remaining_edges:
                    is_connected = False
                    break
            if is_connected:
                valid_new_R.add(candidate)
        
        for vertex in valid_new_R:
            if vertex not in R:
                R.add(vertex)
                changed = True
    
    return L, R

def _edge_count(L, R, remaining_edges):
    """Count edges in biclique within remaining_edges."""
    count = 0
    for l in L:
        for r in R:
            edge_key = (min(l, r), max(l, r))
            if edge_key in remaining_edges:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = P['n']
    edges = P['edges']

    # Build initial adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    remaining_edges = set(edge_set)
    result_bicliques = []

    # Use spectral partitioning to identify natural clusters
    all_vertices = set(range(n))
    clusters = _spectral_partition(all_vertices, adj, remaining_edges, max_depth=3)
    
    # Sort clusters by size (larger first) for better biclique extraction
    clusters.sort(key=len, reverse=True)

    # Process edges in order of cluster density
    # First, try to find bicliques within each cluster
    processed_edges = set()
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        
        # Find edges within this cluster
        cluster_edges = []
        for u, v in remaining_edges:
            if u in cluster and v in cluster:
                if (u, v) not in processed_edges:
                    cluster_edges.append((u, v))
                    processed_edges.add((u, v))
        
        # Try to extract bicliques from cluster edges
        while cluster_edges:
            if time.time() > deadline:
                break
            
            # Pick the first edge as seed
            seed_edge = cluster_edges[0]
            L, R = _find_maximal_biclique_from_seed(seed_edge, adj, remaining_edges)
            
            # Only accept if it covers at least the seed edge
            seed_key = (min(seed_edge[0], seed_edge[1]), max(seed_edge[0], seed_edge[1]))
            if seed_key in remaining_edges:
                # Add biclique
                L_list = [int(x) for x in L]
                R_list = [int(x) for x in R]
                result_bicliques.append([L_list, R_list])
                
                # Remove covered edges
                for l in L:
                    for r in R:
                        edge_key = (min(l, r), max(l, r))
                        if edge_key in remaining_edges:
                            remaining_edges.remove(edge_key)
                            processed_edges.add(edge_key)
                
                # Update cluster_edges
                cluster_edges = [e for e in cluster_edges if e not in processed_edges]
            else:
                # Seed edge already processed, remove it
                cluster_edges.remove(seed_edge)

    # Handle remaining edges not in any cluster
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: add remaining edges as individual bicliques
            for u, v in remaining_edges:
                result_bicliques.append([[int(u)], [int(v)]])
            break

        # Pick an arbitrary edge to start a biclique
        u, v = next(iter(remaining_edges))
        
        # Find maximal biclique
        L, R = _find_maximal_biclique_from_seed((u, v), adj, remaining_edges)
        
        # Add it to result
        L_list = [int(x) for x in L]
        R_list = [int(x) for x in R]
        result_bicliques.append([L_list, R_list])
        
        # Remove all edges of this biclique from remaining_edges
        for l in L:
            for r in R:
                edge_key = (min(l, r), max(l, r))
                if edge_key in remaining_edges:
                    remaining_edges.remove(edge_key)

    return result_bicliques
# EVOLVE-BLOCK-END
