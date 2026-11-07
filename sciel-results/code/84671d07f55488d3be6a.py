import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = list(range(min_vertices))
    best_distinct = 0
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Use degree histogram to maintain distinct count in O(1) per update
    # deg_hist[d] = number of included vertices with degree d
    # distinct = number of d with deg_hist[d] > 0
    deg_array = [-1] * n
    deg_hist = [0] * (n + 1)  # max degree in subgraph of size k is k-1, but n is safe
    included_vertices = []
    distinct_count = 0
    
    def _check_deadline():
        return time.time() - start_time > deadline
    
    def _update_add(v, d):
        """Add vertex v with degree d. Returns new distinct_count."""
        nonlocal distinct_count
        if deg_hist[d] == 0:
            distinct_count += 1
        deg_hist[d] += 1
        deg_array[v] = d
        included_vertices.append(v)
    
    def _update_remove(v, d):
        """Remove vertex v which had degree d. Returns new distinct_count."""
        nonlocal distinct_count
        deg_hist[d] -= 1
        if deg_hist[d] == 0:
            distinct_count -= 1
        deg_array[v] = -1
        included_vertices.pop()
    
    def _increment_deg(v):
        """Increment degree of vertex v. Returns new distinct_count."""
        nonlocal distinct_count
        old_d = deg_array[v]
        new_d = old_d + 1
        deg_hist[old_d] -= 1
        if deg_hist[old_d] == 0:
            distinct_count -= 1
        deg_hist[new_d] += 1
        if deg_hist[new_d] == 1:
            distinct_count += 1
        deg_array[v] = new_d
    
    def _decrement_deg(v):
        """Decrement degree of vertex v. Returns new distinct_count."""
        nonlocal distinct_count
        old_d = deg_array[v]
        new_d = old_d - 1
        deg_hist[old_d] -= 1
        if deg_hist[old_d] == 0:
            distinct_count -= 1
        deg_hist[new_d] += 1
        if deg_hist[new_d] == 1:
            distinct_count += 1
        deg_array[v] = new_d
    
    def dfs(idx, included_mask):
        nonlocal best_subset, best_distinct
        
        if _check_deadline():
            return
            
        cur_size = len(included_vertices)
        remaining = n - idx
        
        # Pruning: Can we reach min_vertices?
        if cur_size + remaining < min_vertices:
            return
            
        # Pruning: Can we beat best_distinct?
        # Max possible distinct degrees = max possible subset size
        if cur_size + remaining < best_distinct:
            return
            
        # If we've considered all vertices
        if idx == n:
            if cur_size >= min_vertices and distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(included_vertices)
            return
            
        # Option 1: Exclude vertex idx
        dfs(idx + 1, included_mask)
        
        # Option 2: Include vertex idx
        deg_idx = bin(adj_mask[idx] & included_mask).count('1')
        
        # Increment degrees of neighbors in included_mask
        neighbors_mask = adj_mask[idx] & included_mask
        temp = neighbors_mask
        changed_neighbors = []
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            changed_neighbors.append(v)
            _increment_deg(v)
            temp ^= lsb
        
        # Add idx
        _update_add(idx, deg_idx)
        
        dfs(idx + 1, included_mask | (1 << idx))
        
        # Backtrack: remove idx and restore neighbor degrees
        _update_remove(idx, deg_idx)
        for v in changed_neighbors:
            _decrement_deg(v)
            
    dfs(0, 0)
    
    return [int(v) for v in best_subset]

construct()
