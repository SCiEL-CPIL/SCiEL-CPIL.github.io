import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Build adjacency as sets for O(1) lookup
    adj = [set() for _ in range(n)]
    
    # Precompute edge positions for quick score calculation
    edge_pos = {}
    for i, (u, v) in enumerate(edges):
        edge_pos[(min(u,v), max(u,v))] = i + 1  # 1-based index
    
    def get_edge_score(cycle):
        """Calculate the max arrival index of edges in the cycle."""
        max_idx = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            key = (min(u,v), max(u,v))
            idx = edge_pos.get(key, 999)
            if idx > max_idx:
                max_idx = idx
        return max_idx
    
    def find_hamilton_cycle(adj, n, deadline):
        """Find Hamilton cycle using backtracking with strong pruning."""
        # Convert to lists for faster iteration
        adj_list = [list(s) for s in adj]
        
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def backtrack(u, depth):
            if time.time() > deadline:
                return None
            
            if depth == n:
                # Check if u connects to start (0)
                if 0 in adj[u]:
                    return path[:]
                return None
            
            # Warnsdorff heuristic: visit neighbors with fewest unvisited neighbors first
            candidates = []
            for v in adj_list[u]:
                if not visited[v]:
                    # Count unvisited neighbors of v
                    unvisited_count = sum(1 for w in adj_list[v] if not visited[w])
                    candidates.append((unvisited_count, v))
            
            if not candidates:
                return None
            
            candidates.sort()
            
            for _, v in candidates:
                visited[v] = True
                path.append(v)
                
                res = backtrack(v, depth + 1)
                if res is not None:
                    return res
                
                path.pop()
                visited[v] = False
            
            return None
        
        return backtrack(0, 1)
    
    # Binary search for the smallest T where a Hamilton cycle exists
    lo, hi = 1, len(edges)
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build graph with edges[0..mid-1]
        adj = [set() for _ in range(n)]
        for i in range(mid):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Quick check: all vertices must have degree >= 2
        if any(len(adj[v]) < 2 for v in range(n)):
            lo = mid + 1
            continue
        
        # Check connectivity (BFS/DFS)
        if len(adj[0]) == 0:
            lo = mid + 1
            continue
        
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack and count < n:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        
        if count < n:
            lo = mid + 1
            continue
        
        # Try to find Hamilton cycle
        cycle = find_hamilton_cycle(adj, n, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search didn't find one, try sequential from low T
    if best_cycle is None:
        for T in range(1, len(edges) + 1):
            if time.time() > deadline:
                break
            
            u, v = edges[T-1]
            # We need to rebuild adj each time or maintain incrementally
            # For simplicity, rebuild
            adj = [set() for _ in range(n)]
            for i in range(T):
                eu, ev = edges[i]
                adj[eu].add(ev)
                adj[ev].add(eu)
            
            if any(len(adj[v]) < 2 for v in range(n)):
                continue
            
            visited = [False] * n
            stack = [0]
            visited[0] = True
            count = 1
            while stack and count < n:
                u = stack.pop()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        stack.append(v)
            
            if count < n:
                continue
            
            cycle = find_hamilton_cycle(adj, n, deadline)
            if cycle is not None:
                best_cycle = cycle
                break
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: return a simple cycle (may have high score)
    return list(range(n))
# EVOLVE-BLOCK-END
