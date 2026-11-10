import time

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Returns a dict with 'edges' (list of undirected edge pairs) and 'cycles'
    (list of vertex-list cycles, one for each length 3..40 in that order).
    """
    t0 = time.time()
    time_limit = 85.0
    
    n_verts = 40
    edge_set = set()
    
    # Base Hamiltonian cycle
    for i in range(n_verts):
        edge_set.add((i, (i + 1) % n_verts))
        
    # Chords from vertex 0
    # These chords ensure pancyclicity
    # k in [2, 20] covers cycle lengths:
    # Small side: k+1 in [3, 21]
    # Large side: 41-k in [21, 39]
    # Union with 40 gives [3, 40]
    for k in range(2, 21):
        if time.time() - t0 > time_limit:
            break
        edge_set.add((0, k))
        
    # Normalize edges to (min, max) and convert to list
    norm_edges = []
    for u, v in edge_set:
        if u < v:
            norm_edges.append((u, v))
        else:
            norm_edges.append((v, u))
            
    # Generate cycles
    cycle_list = []
    
    for target_len in range(3, n_verts + 1):
        if time.time() - t0 > time_limit:
            # Emergency break, though unlikely
            break
            
        if target_len == n_verts:
            # Hamiltonian cycle
            cyc = list(range(n_verts))
        else:
            # Determine which chord to use
            # If target_len <= 21, use chord k = target_len - 1
            # Cycle: 0 -> 1 -> ... -> k -> 0
            # Vertices: 0, 1, ..., k
            # Length of path 0..k is k. Plus chord (k,0) is 1. Total k+1.
            # So k = target_len - 1.
            if target_len <= 21:
                k = target_len - 1
                # k is in [2, 20]
                cyc = list(range(k + 1))
            else:
                # If target_len > 21, use chord k such that 41 - k = target_len
                # k = 41 - target_len
                # k will be in [2, 19] because target_len in [22, 39]
                k = 41 - target_len
                # Cycle: 0 -> 39 -> ... -> k -> 0
                # Vertices: 0, 39, 38, ..., k
                # Path from 0 to k via 39: 0->39->...->k
                # This is the "long way" around the Hamiltonian cycle.
                # Length is 41-k.
                # Construct list: 0, then range(39, k-1, -1)
                cyc = [0] + list(range(39, k - 1, -1))
                
        cycle_list.append(cyc)
        
    # Ensure we have all cycles if deadline was hit early (fallback)
    while len(cycle_list) < n_verts - 2:
        L = len(cycle_list) + 3
        if L <= 21:
            k = L - 1
            cyc = list(range(k + 1))
        else:
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycle_list.append(cyc)
        
    return {'edges': norm_edges, 'cycles': cycle_list}
