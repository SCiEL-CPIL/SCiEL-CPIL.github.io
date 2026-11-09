import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']
TOTAL_EDGES = N * (N - 1) // 2

# Precompute edge index lookup (one-based)
edge_index = {}
for i, e in enumerate(EDGES):
    u, v = e
    if u > v:
        u, v = v, u
    edge_index[(u, v)] = i + 1


def build_adjacency(threshold):
    """Build adjacency bitsets for edges with index <= threshold."""
    adj = [0] * N
    for i in range(threshold):
        u, v = EDGES[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def find_hamilton_cycle_fast(adj, deadline, max_backtracks=50000):
    """
    Find Hamiltonian cycle using greedy with limited backtracking.
    Much faster than full DFS.
    """
    full_mask = (1 << N) - 1
    best_path = None
    best_score = float('inf')
    
    def popcount(x):
        return bin(x).count('1')
    
    def try_greedy(start_vertex, adj, deadline, max_backtracks):
        """Try greedy construction from start_vertex with limited backtracking."""
        if time.time() > deadline:
            return None
        
        best_local = None
        best_local_score = float('inf')
        
        # Try different orderings of initial neighbors
        init_neighbors = adj[start_vertex]
        if init_neighbors == 0:
            return None
        
        # Get neighbor list sorted by degree (Warnsdorff)
        nbr_list = []
        temp = init_neighbors
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(adj[v])
            nbr_list.append((deg, v))
            temp ^= lsb
        nbr_list.sort()
        
        for _, v_first in nbr_list:
            if time.time() > deadline:
                break
            
            visited = (1 << start_vertex) | (1 << v_first)
            path = [start_vertex, v_first]
            current = v_first
            backtracks = 0
            path_stack = [(current, visited, path[:])]
            
            while path_stack and backtracks < max_backtracks:
                if time.time() > deadline:
                    break
                
                node, vis, pth = path_stack[-1]
                
                if len(pth) == N:
                    # Check if we can close the cycle
                    if adj[node] & (1 << start_vertex):
                        score = max(edge_index.get((min(pth[0], pth[-1]), max(pth[0], pth[-1])), float('inf')) if pth[0] != pth[-1] else 0,
                                   max(edge_index.get((min(pth[i], pth[i+1]), max(pth[i], pth[i+1])), float('inf')) for i in range(N-1)))
                        if score < best_local_score:
                            best_local_score = score
                            best_local = pth[:]
                        path_stack.pop()
                        continue
                
                # Get unvisited neighbors
                nbrs = adj[node] & ~vis
                if nbrs == 0:
                    path_stack.pop()
                    backtracks += 1
                    continue
                
                # Build candidates sorted by degree (Warnsdorff)
                candidates = []
                temp = nbrs
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    # Dead-end check: does v have any unvisited neighbor?
                    new_vis = vis | (1 << v)
                    if new_vis != full_mask:
                        remaining = (~new_vis) & full_mask
                        if (adj[v] & remaining) == 0:
                            temp ^= lsb
                            continue
                    deg = popcount(adj[v] & ~vis)
                    candidates.append((deg, v))
                    temp ^= lsb
                
                if not candidates:
                    path_stack.pop()
                    backtracks += 1
                    continue
                
                # Sort by degree (prefer lower degree - Warnsdorff)
                candidates.sort()
                
                # Try candidates in order
                found_next = False
                for _, v in candidates:
                    if time.time() > deadline:
                        break
                    
                    new_vis = vis | (1 << v)
                    # Quick connectivity check for remaining nodes
                    remaining = (~new_vis) & full_mask
                    if remaining:
                        # Check if remaining graph is connected
                        first_rem = (remaining & -remaining).bit_length() - 1
                        seen = 1 << first_rem
                        queue = [first_rem]
                        while queue:
                            cu = queue.pop(0)
                            nb = adj[cu] & remaining
                            while nb:
                                lsb2 = nb & -nb
                                w = lsb2.bit_length() - 1
                                if not (seen & (1 << w)):
                                    seen |= (1 << w)
                                    queue.append(w)
                                nb ^= lsb2
                        if seen != remaining:
                            continue  # Would disconnect
                    
                    path_stack.append((v, new_vis, pth + [v]))
                    found_next = True
                    break
                
                if not found_next:
                    path_stack.pop()
                    backtracks += 1
        
        return best_local
    
    # Try from each start vertex
    for start_vertex in range(N):
        if time.time() > deadline:
            break
        
        result = try_greedy(start_vertex, adj, deadline, max_backtracks)
        if result is not None:
            # Calculate score
            score = 0
            for i in range(N):
                u, v = result[i], result[(i+1) % N]
                idx = edge_index.get((min(u,v), max(u,v)), float('inf'))
                score = max(score, idx)
            if score < best_score:
                best_score = score
                best_path = result
    
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Strategy: Start with a reasonable threshold and increase if needed
    # For 24 vertices, a Hamilton cycle needs at least 24 edges
    # Try thresholds in increasing order with exponential steps
    
    best_path = None
    best_score = float('inf')
    
    # First, try to find a cycle with a moderate threshold
    # Start with threshold around 50-100 and adjust
    thresholds_to_try = [40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    
    for threshold in thresholds_to_try:
        if time.time() > deadline:
            break
        if threshold > TOTAL_EDGES:
            break
        
        adj = build_adjacency(threshold)
        
        # Check if graph is connected and has minimum degree >= 2
        # Quick check: every vertex must have degree >= 2
        min_deg = min(bin(adj[v]).count('1') for v in range(N))
        if min_deg < 2:
            continue
        
        path = find_hamilton_cycle_fast(adj, deadline, max_backtracks=20000)
        
        if path is not None:
            # Calculate score
            score = 0
            for i in range(N):
                u, v = path[i], path[(i+1) % N]
                idx = edge_index.get((min(u,v), max(u,v)), float('inf'))
                score = max(score, idx)
            
            if score < best_score:
                best_score = score
                best_path = path
            
            # If we found a cycle, try to improve with lower thresholds
            # But first check if we have time
            if time.time() > deadline - 10:
                break
    
    # If no path found, try with full graph
    if best_path is None:
        adj_full = build_adjacency(TOTAL_EDGES)
        path = find_hamilton_cycle_fast(adj_full, deadline, max_backtracks=50000)
        if path is not None:
            best_path = path
    
    # Fallback: return trivial cycle
    if best_path is None:
        return list(range(N))
    
    return best_path
# EVOLVE-BLOCK-END
