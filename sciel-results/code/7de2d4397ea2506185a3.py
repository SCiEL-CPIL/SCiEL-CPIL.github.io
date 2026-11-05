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
    
    best_path = []
    best_len = 0
    deadline = time.time() + 55.0  # Leave 5s margin for 60s limit
    
    # Helper: count unvisited neighbors
    def count_unvisited(node, visited_mask):
        return bin(adj[node] & ~visited_mask).count('1')
    
    # Helper: check if removing vertex v disconnects the remaining unvisited graph from current endpoint u
    # Returns True if v is an articulation point that would isolate some unvisited vertices
    def is_articulation_point(v, u, visited_mask):
        # Find all unvisited vertices reachable from u without going through v
        # If any unvisited vertex is not reachable, v is an articulation point
        unvisited = ~visited_mask
        # Remove v from consideration
        unvisited_no_v = unvisited & ~(1 << v)
        
        # BFS from u, avoiding v
        if unvisited_no_v == 0:
            return False
        
        reachable = 0
        stack = [u]
        seen = 1 << u
        while stack:
            node = stack.pop()
            mask = adj[node] & unvisited_no_v & ~seen
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                if w == v:
                    mask ^= lsb
                    continue
                seen |= (1 << w)
                stack.append(w)
                mask ^= lsb
            reachable |= seen
        
        # Check if all unvisited (except v) are reachable
        if (unvisited_no_v & ~reachable) != 0:
            return True
        return False
    
    # DFS with enhanced pruning
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
            return True  # Dead end, backtrack
        
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
        neighbors.sort(key=lambda v: count_unvisited(v, visited_mask))
        
        for v in neighbors:
            new_mask = visited_mask | (1 << v)
            path.append(v)
            dfs(v, path, new_mask)
            path.pop()
            
            if best_len == n:
                return True
                
            if time.time() > deadline:
                return False
                
        return True
    
    # Try starting from vertices in the largest component
    start_vertices = sorted(components[0], key=lambda x: degrees[x])
    
    for start in start_vertices:
        if time.time() > deadline:
            break
            
        if adj[start] == 0:
            continue
            
        path = [start]
        visited_mask = 1 << start
        dfs(start, path, visited_mask)
        
        if time.time() > deadline:
            break

    if not best_path:
        return [0]
        
    return best_path

# EVOLVE-BLOCK-END
