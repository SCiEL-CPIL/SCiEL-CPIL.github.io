import sys
import time
import random

# Set recursion limit
sys.setrecursionlimit(10000)

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    weight = [[0] * n for _ in range(n)]
    for i, (u, v) in enumerate(P['edge_order']):
        weight[u][v] = i + 1
        weight[v][u] = i + 1

    # Precompute sorted edge weights to help with bounds?
    # We need to find min K such that G_K has a Hamiltonian cycle.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    # Binary search bounds
    # Min edges for cycle is n.
    # Max is n*(n-1)//2
    low = n
    high = n * (n - 1) // 2
    best_k = high
    best_cycle = None

    def check_cycle(K):
        # Build adjacency masks for G_K
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if u != v and weight[u][v] <= K:
                    mask |= (1 << v)
            adj_masks[u] = mask
            
        # Pre-check: Degree >= 2
        for u in range(n):
            if bin(adj_masks[u]).count('1') < 2:
                return None

        # Pre-check: Connectivity (BFS/DFS on undirected graph)
        # If not connected, no cycle.
        visited = 1 # Start at 0
        stack = [0]
        while stack:
            u = stack.pop()
            neighbors = adj_masks[u] & ~visited
            if neighbors:
                # Get lowest set bit
                lsb = neighbors & -neighbors
                v = lsb.bit_length() - 1
                visited |= (1 << v)
                stack.append(v)
        if visited != (1 << n) - 1:
            return None

        # DFS with Warnsdorff
        # State: visited_mask, last_vertex
        # We fix start vertex to 0 to reduce symmetry? 
        # Hamiltonian cycle is symmetric, but fixing start is valid.
        
        # Heuristic: Start with vertex with min degree? 
        # Or just 0. Let's try min degree for better pruning.
        start_vertex = min(range(n), key=lambda x: bin(adj_masks[x]).count('1'))
        
        visited_mask = 1 << start_vertex
        path = [start_vertex]
        
        def dfs(last, visited_mask, path):
            if time.time() > deadline:
                return False # Timeout, treat as failure for this K
            
            if len(path) == n:
                # Check if cycle closes
                if (1 << start_vertex) & adj_masks[last]:
                    return True
                return False
            
            last = path[-1]
            candidates = adj_masks[last] & ~visited_mask
            
            if not candidates:
                return False
            
            # Warnsdorff: Sort candidates by number of onward moves
            # onward moves = (adj_masks[v] & ~visited_mask).bit_count()
            # Note: We must also ensure that the remaining unvisited vertices 
            # can still be connected.
            
            cand_list = []
            temp_mask = candidates
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                onward = (adj_masks[v] & ~visited_mask).bit_count()
                cand_list.append((onward, v))
                temp_mask ^= lsb
            
            cand_list.sort()
            
            for onward, v in cand_list:
                # Pruning: 
                # If we pick v, are we leaving any unvisited vertex w 
                # with no available neighbors (except possibly v)?
                # This check is expensive. 
                # Simple check: If any unvisited vertex (excluding v) 
                # has (adj_masks[w] & ~visited_mask & ~(1<<v)) == 0, 
                # and it's not connected to v, it's isolated.
                # Actually, if (adj_masks[w] & ~visited_mask) has only 1 bit set, 
                # and that bit is NOT v, then w is dead.
                # If that bit IS v, then w MUST be visited next after v.
                
                # Let's do a light check:
                # For each unvisited w != v:
                # avail_w = adj_masks[w] & ~visited_mask
                # if avail_w == 0: return False (shouldn't happen if we check degree)
                # if avail_w has 1 bit:
                #    if that bit is not v: return False (w is isolated from v)
                #    if that bit is v: 
                #       then v must go to w next. 
                #       This is a constraint.
                
                # To keep it fast, we only do this if the number of unvisited vertices is small?
                # Or just rely on Warnsdorff.
                
                new_visited = visited_mask | (1 << v)
                path.append(v)
                if dfs(v, new_visited, path):
                    return True
                path.pop()
            
            return False

        if dfs(start_vertex, visited_mask, path):
            return path
        return None

    # Binary Search
    while low <= high:
        if time.time() > deadline - 5.0:
            break # Safety margin
            
        mid = (low + high) // 2
        cycle = check_cycle(mid)
        if cycle:
            best_cycle = cycle
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1

    if best_cycle:
        return best_cycle
    else:
        # Fallback: Return a cycle from the full graph (K = max)
        # This should not happen, but just in case
        return list(range(n))
