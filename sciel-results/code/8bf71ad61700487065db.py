import random
import time
import sys

def construct():
    n = 24
    # Generate and shuffle edges with pinned seed
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    r = random.Random(614)
    r.shuffle(edges)
    
    # Build edge arrival index: edge_index[(u,v)] = 0-based index in shuffled list
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def get_arrival(u, v):
        return edge_index[(u, v)]
    
    def has_hamiltonian_cycle(adj_matrix):
        """Check if Hamilton cycle exists using optimized backtracking with strong pruning."""
        # Check degree constraint
        for i in range(n):
            if bin(adj_matrix[i]).count('1') < 2:
                return None
        
        # Use bitmask-based DFS with strong pruning
        full_mask = (1 << n) - 1
        path = [0]
        visited_mask = 1
        
        # Precompute neighbor lists as lists of bits for faster iteration
        def get_neighbors(mask):
            """Return list of (vertex, bit) for set bits in mask, ordered by degree (ascending)."""
            result = []
            m = mask
            while m:
                b = m & (-m)
                v = b.bit_length() - 1
                m ^= b
                result.append((bin(adj_matrix[v] & full_mask).count('1'), v, b))
            result.sort()
            return result
        
        def is_connected(mask, start):
            """Check connectivity of vertices in mask using BFS."""
            if mask == 0:
                return True
            visited = 1 << start
            stack = [start]
            while stack:
                u = stack.pop()
                nbrs = adj_matrix[u] & mask
                while nbrs:
                    b = nbrs & (-nbrs)
                    nbrs ^= b
                    if not (visited & b):
                        visited |= b
                        stack.append(b.bit_length() - 1)
            return visited == mask
        
        def dfs(u, visited_mask, path):
            if len(path) == n:
                if adj_matrix[u] & 1:
                    return path[:]
                return None
            
            remaining = n - len(path)
            unvisited = full_mask & ~visited_mask
            
            # Available next vertices
            avail = adj_matrix[u] & unvisited
            if remaining == 1:
                avail &= adj_matrix[0]
            
            if avail == 0:
                return None
            
            # Pruning: check connectivity of unvisited vertices + {u, 0}
            # The unvisited vertices must be connected to each other through the remaining graph
            # Actually, a simpler and effective prune: every unvisited vertex must have
            # at least one neighbor in (unvisited ∪ {0}) if it's not the last, or in {0} if last
            # Let's check: for each unvisited vertex w, does it have a neighbor in unvisited or 0?
            if remaining > 1:
                for w in range(n):
                    w_bit = 1 << w
                    if w_bit & unvisited:
                        # w must be reachable: needs neighbor in unvisited or in {0} (for closing)
                        # But w also needs to be able to reach the rest, so check it has at least
                        # one neighbor in (unvisited ∪ {0})
                        w_nbrs = adj_matrix[w] & (unvisited | 1)  # 1 is vertex 0
                        if w_nbrs == 0:
                            return None
            else:
                # Only one vertex left, it must connect to 0
                if not (avail & adj_matrix[0]):
                    return None
            
            # MRV: try neighbors with fewest forward options first
            nbrs = get_neighbors(avail)
            # Filter: if remaining==1, only consider neighbors that connect to 0
            if remaining == 1:
                nbrs = [(d, v, b) for d, v, b in nbrs if adj_matrix[v] & 1]
            
            for _, v, v_bit in nbrs:
                new_mask = visited_mask | v_bit
                res = dfs(v, new_mask, path + [v])
                if res:
                    return res
            
            return None
        
        return dfs(0, visited_mask, path)
    
    # Build adjacency incrementally and use binary search on K
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    # First, find a valid upper bound by scanning from a reasonable start
    # The minimum possible K is n (24 edges for a cycle on 24 vertices)
    # Let's start scanning but with binary search strategy
    
    # Build adjacency for a given K
    def build_adj(K):
        """Build adjacency bitmasks using first K edges."""
        adj = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Check if Hamilton cycle exists at K
    def check_K(K):
        if K < n:
            return None
        adj = build_adj(K)
        # Quick degree check
        for i in range(n):
            if bin(adj[i]).count('1') < 2:
                return None
        return has_hamiltonian_cycle(adj)
    
    # Binary search for minimum K
    # Lower bound: n (need at least n edges for a cycle)
    # Upper bound: len(edges) (complete graph always has Hamilton cycle)
    lo = n
    hi = len(edges)
    
    # First, verify that hi works (sanity check, should always be true)
    # Don't spend time on this, just proceed with binary search
    
    best_cycle = None
    best_K = hi
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = check_K(mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try a few more K values near the end
    for K in range(hi + 1, min(hi + 20, len(edges) + 1)):
        if time.time() > deadline:
            break
        cycle = check_K(K)
        if cycle is not None:
            return cycle
    
    # Ultimate fallback
    return list(range(n))
