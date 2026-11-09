import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix for O(1) lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    # Helper to check if v is connected to all/none in mask
    def connected_to_all(v, mask):
        m = mask
        while m:
            u = (m & -m).bit_length() - 1
            if not adj[v][u]:
                return False
            m &= m - 1
        return True
        
    def connected_to_none(v, mask):
        m = mask
        while m:
            u = (m & -m).bit_length() - 1
            if adj[v][u]:
                return False
            m &= m - 1
        return True

    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and connected_to_all(v, rest)
        is_independent[mask] = is_independent[rest] and connected_to_none(v, rest)
        
    # Collect valid masks by vertex
    masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                masks_by_vertex[v].append(mask)
                m &= m - 1
                
    # Sort masks by size descending
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Convert masks to vertex lists for quick extraction
    mask_to_vertices = {}
    for mask in range(size):
        if mask == 0:
            mask_to_vertices[mask] = []
        else:
            verts = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                verts.append(u)
                m &= m - 1
            mask_to_vertices[mask] = verts
    
    best_parts = n  # Max parts is n (singletons)
    best_partition = None
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    def dfs(remaining_mask, current_depth, current_parts):
        nonlocal best_parts, best_partition
        
        if time.time() > deadline:
            return
            
        if current_depth >= best_parts:
            return
            
        if remaining_mask == 0:
            if current_depth < best_parts:
                best_parts = current_depth
                best_partition = current_parts[:]
            return
            
        # Pick lowest set bit
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        
        # Try all valid masks containing v that are subsets of remaining
        for S in masks_by_vertex[v]:
            if S & remaining_mask == S:
                new_remaining = remaining_mask ^ S
                part_vertices = mask_to_vertices[S]
                current_parts.append(part_vertices)
                
                dfs(new_remaining, current_depth + 1, current_parts)
                
                current_parts.pop()
                
                if time.time() > deadline:
                    return

    dfs(full_mask, 0, [])
    
    # Fallback if something went wrong
    if best_partition is None:
        best_partition = [[i] for i in range(n)]
        
    # Ensure types are correct: list of lists of int
    result = []
    for part in best_partition:
        result.append([int(v) for v in part])
    return result
