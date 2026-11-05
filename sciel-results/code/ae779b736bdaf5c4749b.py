import random
import time

def construct():
    n = 24
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    r = random.Random(614)
    r.shuffle(edges)
    
    # Precompute edge position for scoring
    edge_pos = {}
    for idx, (u, v) in enumerate(edges):
        edge_pos[(min(u,v), max(u,v))] = idx + 1  # 1-based
    
    def cycle_score(cycle):
        """Compute the max 1-based arrival index of edges in the cycle."""
        max_idx = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            key = (min(u, v), max(u, v))
            max_idx = max(max_idx, edge_pos.get(key, float('inf')))
        return max_idx
    
    # Build adjacency for a given K
    def build_adj(K):
        adj = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Check basic feasibility: degree >= 2 for all, connectivity
    def feasible(adj):
        for i in range(n):
            if bin(adj[i]).count('1') < 2:
                return False
        # BFS connectivity
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            nb = adj[u]
            while nb:
                lsb = nb & (-nb)
                v = lsb.bit_length() - 1
                nb ^= lsb
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        return count == n
    
    # Find Hamiltonian cycle in graph with adjacency adj
    # Returns the cycle as a list of vertices, or None
    def find_cycle(adj, deadline=None):
        if not feasible(adj):
            return None
        
        full_mask = (1 << n) - 1
        path = [0]
        
        # Precompute degree in unvisited for faster Warnsdorff
        # Use iterative DFS with explicit stack to avoid recursion overhead
        # But n=24, recursion is fine
        
        def dfs(u, visited_mask):
            if deadline is not None and time.time() > deadline:
                return False
            
            if len(path) == n:
                return (adj[u] & 1) != 0
            
            unvisited = full_mask ^ visited_mask
            candidates = adj[u] & unvisited
            
            if candidates == 0:
                return False
            
            # Warnsdorff: order by degree in unvisited (ascending)
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg_v = bin(adj[v] & unvisited).count('1')
                cand_list.append((deg_v, v))
            cand_list.sort()
            
            for _, v in cand_list:
                path.append(v)
                if dfs(v, visited_mask | (1 << v)):
                    return True
                path.pop()
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None
    
    # Time management
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Strategy: Binary search for minimum K, then find cycle
    # But also keep track of best cycle found at any K
    
    best_cycle = None
    best_score = float('inf')
    
    # First, ensure we have a valid cycle from the full graph as fallback
    # Do this quickly with a relaxed deadline
    fallback_deadline = start_time + 2
    adj_full = build_adj(len(edges))
    fallback_cycle = find_cycle(adj_full, fallback_deadline)
    if fallback_cycle is not None:
        best_cycle = fallback_cycle
        best_score = cycle_score(fallback_cycle)
    
    # Binary search for minimum K
    low = n
    high = len(edges)
    k_min = None
    
    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        adj = build_adj(mid)
        if not feasible(adj):
            low = mid + 1
            continue
        
        # Try to find a cycle
        cycle = find_cycle(adj, deadline)
        if cycle is not None:
            k_min = mid
            score = cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
    
    # If we found a cycle at k_min, try to verify it's truly minimal
    # by checking k_min - 1 (but we already did binary search, so this is redundant)
    
    # If best_cycle is still None (shouldn't happen), use fallback
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle

if __name__ == "__main__":
    pass
