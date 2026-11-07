import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(tuple(sorted((u, v))))
    
    # Forbidden edges
    forbidden = set()
    
    def find_hamiltonian_cycle_dp():
        """Find a Hamiltonian cycle using bitmask DP (Held-Karp style)."""
        start = 0
        full_mask = (1 << n) - 1
        
        # dp[mask][u] = True if there's a path from start to u visiting exactly mask
        # We use a dict-based approach for memory efficiency
        # State: (mask, last_vertex) -> predecessor
        from collections import defaultdict
        dp = {}
        dp[(1 << start, start)] = None  # base case
        
        # Iterate over masks by size
        for size in range(1, n):
            # For each mask of this size, check transitions
            for mask in range(1, 1 << n):
                if bin(mask).count('1') != size:
                    continue
                for u in range(n):
                    if not (mask & (1 << u)):
                        continue
                    key = (mask, u)
                    if key not in dp:
                        continue
                    # Try extending to v
                    for v in adj[u]:
                        if mask & (1 << v):
                            continue
                        # Check edge u-v is available
                        e = tuple(sorted((u, v)))
                        if e in forbidden:
                            continue
                        new_mask = mask | (1 << v)
                        new_key = (new_mask, v)
                        if new_key not in dp:
                            dp[new_key] = key  # store predecessor
        
        # Check if we can complete the cycle
        for u in range(n):
            if u == start:
                continue
            key = (full_mask, u)
            if key not in dp:
                continue
            e = tuple(sorted((u, start)))
            if e in forbidden:
                continue
            if e not in edge_set:
                continue
            # Reconstruct path
            path = [u]
            cur = key
            while dp[cur] is not None:
                cur = dp[cur]
                path.append(cur[1])
            path.reverse()
            return path
        
        return None

    def find_cycle_backtracking():
        """Find a Hamiltonian cycle using backtracking with strong pruning."""
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        # Precompute available neighbors (not forbidden)
        def avail_neighbors(u, visited_mask):
            result = []
            for v in adj[u]:
                if visited_mask & (1 << v):
                    continue
                e = tuple(sorted((u, v)))
                if e in forbidden:
                    continue
                result.append(v)
            return result
        
        def dfs(u, depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                e = tuple(sorted((u, start)))
                if e not in forbidden and e in edge_set:
                    return True
                return False
            
            # Get available neighbors, sorted by fewest onward options
            neighbors = []
            for v in adj[u]:
                if visited[v]:
                    continue
                e = tuple(sorted((u, v)))
                if e in forbidden:
                    continue
                if e not in edge_set:
                    continue
                # Count onward options for v
                opts = 0
                for w in adj[v]:
                    if not visited[w] and w != u:
                        ew = tuple(sorted((v, w)))
                        if ew not in forbidden and ew in edge_set:
                            opts += 1
                neighbors.append((opts, v))
            
            neighbors.sort()
            
            for _, v in neighbors:
                visited[v] = True
                path.append(v)
                
                if dfs(v, depth + 1):
                    return True
                
                visited[v] = False
                path.pop()
            
            return False
        
        if dfs(start, 1):
            return path[:]
        return None

    def try_cycle_with_start(start_vertex):
        """Try to find a Hamiltonian cycle starting from a specific vertex."""
        path = [start_vertex]
        visited = [False] * n
        visited[start_vertex] = True
        
        def dfs(u, depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                e = tuple(sorted((u, start_vertex)))
                if e not in forbidden and e in edge_set:
                    return True
                return False
            
            neighbors = []
            for v in adj[u]:
                if visited[v]:
                    continue
                e = tuple(sorted((u, v)))
                if e in forbidden:
                    continue
                if e not in edge_set:
                    continue
                opts = 0
                for w in adj[v]:
                    if not visited[w] and w != u:
                        ew = tuple(sorted((v, w)))
                        if ew not in forbidden and ew in edge_set:
                            opts += 1
                neighbors.append((opts, v))
            
            neighbors.sort()
            
            for _, v in neighbors:
                visited[v] = True
                path.append(v)
                
                if dfs(v, depth + 1):
                    return True
                
                visited[v] = False
                path.pop()
            
            return False
        
        if dfs(start_vertex, 1):
            return path[:]
        return None

    cycles = []
    max_possible = 3  # floor(70/21)
    
    for i in range(max_possible):
        if time.time() > deadline - 10:
            break
        
        # Edge budget check: need at least n more non-forbidden edges
        remaining_edges = len(edge_set) - len(forbidden)
        if remaining_edges < n:
            break
        
        cycle = None
        
        # Try DP first (more reliable but potentially slower)
        if time.time() < deadline - 40:
            cycle = find_hamiltonian_cycle_dp()
        
        if cycle is None:
            # Try backtracking from vertex 0
            cycle = find_cycle_backtracking()
        
        if cycle is None:
            # Try different starting vertices
            for sv in range(1, n):
                if time.time() > deadline - 10:
                    break
                cycle = try_cycle_with_start(sv)
                if cycle is not None:
                    break
        
        if cycle is None:
            break
        
        # Validate cycle
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e not in edge_set or e in forbidden:
                valid = False
                break
        if not valid:
            continue
        
        cycles.append(cycle)
        
        # Mark edges as forbidden
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            forbidden.add(tuple(sorted((u, v))))
    
    # Fallback
    if not cycles:
        seed_cycle = list(range(n))
        valid = True
        for j in range(n):
            u = seed_cycle[j]
            v = seed_cycle[(j + 1) % n]
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                valid = False
                break
        if valid:
            cycles.append(seed_cycle)
    
    if not cycles:
        return []
    
    return cycles
# EVOLVE-BLOCK-END
