import random
import time

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # Generate the pinned edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    r = random.Random(614)
    r.shuffle(edges)
    
    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # Precompute adjacency lists sorted by edge arrival index
    # adj[u] = list of (v, idx) sorted by idx
    adj_raw = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj_raw[u].append((v, idx))
        adj_raw[v].append((u, idx))
    
    for i in range(n):
        adj_raw[i].sort(key=lambda x: x[1])
    
    # For fast lookup of max index in a cycle, we need the index of each edge.
    # We can just use edge_to_idx.
    
    # We want to find a Hamilton cycle C that minimizes max_{e in C} idx(e).
    # This is equivalent to finding the smallest K such that the graph G_K (edges with idx < K) contains a Hamilton cycle.
    # Then we return a cycle in G_K.
    
    # Since the graph is dense quickly, K is likely small.
    # Let's try to find the minimum K via binary search, then extract a cycle.
    
    # Helper: Check if G_K has a Hamilton cycle.
    # We can use a simple DFS with pruning for n=24.
    # To make it fast, we precompute bitmasks for adjacency in G_K.
    
    def has_hamiltonian_cycle(K):
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Quick check: min degree
        for i in range(n):
            if adj_mask[i].bit_count() < 2:
                return False
        
        # Connectivity check
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            m = adj_mask[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return False
        
        # DFS for Hamiltonian path from 0 to 0 (cycle)
        # State: (current, visited_mask)
        # Optimization: Use Warnsdorff's rule (fewest options first)
        
        # We will do a recursive DFS.
        # To avoid recursion limits and overhead, we can use a stack, but recursion is fine for depth 24.
        # However, we need to be careful with time.
        
        def dfs(u, visited_mask, path):
            if len(path) == n:
                # Check if last node connects to 0
                return (adj_mask[u] & 1) != 0
            
            candidates = adj_mask[u] & (~visited_mask)
            
            # Sort candidates by degree in the remaining graph (Warnsdorff)
            # We need the degree of each candidate in the subgraph induced by unvisited nodes.
            # deg(v) = popcount(adj_mask[v] & ~visited_mask)
            # Note: v is not in visited_mask yet.
            
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj_mask[v] & ~visited_mask).bit_count()
                cand_list.append((deg, v))
            
            cand_list.sort()
            
            for deg, v in cand_list:
                path.append(v)
                if dfs(v, visited_mask | (1 << v), path):
                    return True
                path.pop()
            
            return False
        
        path = [0]
        return dfs(0, 1, path)
    
    def find_hamiltonian_cycle(K):
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        for i in range(n):
            if adj_mask[i].bit_count() < 2:
                return None
        
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            m = adj_mask[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return None
        
        path = [0]
        
        def dfs(u, visited_mask):
            if len(path) == n:
                if (adj_mask[u] & 1) != 0:
                    return True
                return False
            
            candidates = adj_mask[u] & (~visited_mask)
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj_mask[v] & ~visited_mask).bit_count()
                cand_list.append((deg, v))
            
            cand_list.sort()
            
            for deg, v in cand_list:
                path.append(v)
                if dfs(v, visited_mask | (1 << v)):
                    return True
                path.pop()
            
            return False
        
        if dfs(0, 1):
            return path
        return None
    
    # Binary search for minimum K
    low = n
    high = len(edges)
    k_min = len(edges)
    
    # We know K=276 (full graph) works.
    # Let's binary search.
    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        if has_hamiltonian_cycle(mid):
            k_min = mid
            high = mid - 1
        else:
            low = mid + 1
    
    # Now find the cycle for k_min
    if time.time() > deadline:
        # Fallback: try a larger K
        k_try = min(len(edges), k_min + 10)
        cycle = find_hamiltonian_cycle(k_try)
        if cycle:
            return cycle
        # If that fails, return a greedy cycle on full graph
        # Greedy: Start at 0, always go to unvisited neighbor with smallest edge index, 
        # ensuring we can still reach 0 at the end? Hard.
        # Just return a random permutation? No, must be valid.
        # Let's just try find_hamiltonian_cycle(len(edges))
        cycle = find_hamiltonian_cycle(len(edges))
        if cycle:
            return cycle
        return list(range(n))
    
    cycle = find_hamiltonian_cycle(k_min)
    if cycle:
        return cycle
    
    # If find_hamiltonian_cycle failed for k_min (shouldn't happen if has_hamiltonian_cycle succeeded, 
    # unless there's a bug or timeout in the second call), try a slightly larger K.
    # Or just try full graph.
    if time.time() < deadline:
        cycle = find_hamiltonian_cycle(len(edges))
        if cycle:
            return cycle
    
    return list(range(n))

if __name__ == "__main__":
    pass
