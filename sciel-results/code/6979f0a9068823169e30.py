import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Pre-compute degrees for initial sorting
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Find connected components to focus search on the largest one
    # If the graph is disconnected, the longest path is in the largest component
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                # Get neighbors
                mask = adj[node]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
                    mask ^= lsb
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    # We will search in the largest component. 
    # If the largest component is small, we might check others, but usually the longest path is in the largest CC.
    # Let's just search the largest one first. If time permits, we can check others.
    
    best_path = []
    best_len = 0
    deadline = time.time() + 50.0  # Leave 10s margin for 60s limit
    
    # Helper to get unvisited neighbors count for a node given a visited mask
    def count_unvisited_neighbors(node, visited_mask):
        avail = adj[node] & ~visited_mask
        return bin(avail).count('1')

    # DFS with heuristic ordering
    def dfs(u, path, visited_mask):
        nonlocal best_path, best_len
        
        # Check time
        if time.time() > deadline:
            return False
        
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Get available neighbors
        avail = adj[u] & ~visited_mask
        if avail == 0:
            return True # Dead end, backtrack
        
        # Get list of available neighbors
        neighbors = []
        m = avail
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            neighbors.append(v)
            m ^= lsb
        
        # Heuristic: Sort by number of unvisited neighbors (Warnsdorff's rule)
        # Nodes with fewer options are more "critical" and should be visited earlier
        # to avoid trapping the path end.
        neighbors.sort(key=lambda v: count_unvisited_neighbors(v, visited_mask))
        
        for v in neighbors:
            new_mask = visited_mask | (1 << v)
            path.append(v)
            dfs(v, path, new_mask)
            path.pop()
            
            # If we found a Hamiltonian path (or very long), we can stop early?
            # No, we want the global max, but if we found n, we can stop.
            if best_len == n:
                return True
                
            if time.time() > deadline:
                return False
                
        return True

    # Try starting from vertices in the largest component
    # Order starts by degree (lowest first) as endpoints of long paths often have low degree
    start_vertices = sorted(components[0], key=lambda x: degrees[x])
    
    for start in start_vertices:
        if time.time() > deadline:
            break
            
        # If we already have a path longer than the size of this component, we can't beat it
        # But we are in the largest component, so this check is less relevant unless we check smaller ones later.
        
        # Initial heuristic check: if the start node has 0 neighbors, skip
        if adj[start] == 0:
            continue
            
        path = [start]
        visited_mask = 1 << start
        dfs(start, path, visited_mask)
        
        # If we found a very long path, maybe we don't need to try more starts?
        # Hard to say. We'll let the time limit handle it.
        
        # Optimization: If best_len is close to component size, we might stop.
        # But for now, just iterate.

    # If the largest component didn't yield a good result, or if we have time, check others?
    # For now, the largest component is the most likely candidate.
    
    if not best_path:
        # Fallback: return any single vertex if no edges or something went wrong
        return [0]
        
    return best_path

# EVOLVE-BLOCK-END
