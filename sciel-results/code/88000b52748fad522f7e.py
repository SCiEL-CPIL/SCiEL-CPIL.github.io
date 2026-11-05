"""Find a Hamilton cycle minimizing the max edge arrival index in the pinned edge order."""
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)  # 276

# Precompute adjacency for each prefix K
# adj_mask[v] = bitmask of neighbors connected via edges with index <= K
# We'll build incrementally.

# For binary search, we need to check: does G_K have a Hamilton cycle?
# G_K has edges from edge_order[0..K-1] (first K edges, 1-based index <= K)

def build_adj_masks(K):
    """Build adjacency bitmasks for the graph with first K edges."""
    adj = [0] * N
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def has_hamilton_cycle(adj, start=0):
    """Check if graph with adjacency bitmasks has a Hamilton cycle through all N vertices.
    Returns the cycle as a list of vertices, or None."""
    n = N
    start = 0
    start_mask = 1 << start
    # We need to return to start, so the last vertex must be adjacent to start.
    
    # Use iterative DFS with memoization on (current, visited_mask)
    # But 24 vertices -> 2^24 states is too many. Use pruning.
    
    # Better: backtracking with strong pruning
    # Pruning: if the set of unvisited vertices has a component that can't connect back, prune.
    
    # Simpler approach: use the fact that we need a cycle, so each vertex must have degree >= 2 in the cycle.
    # In G_K, if any vertex has degree < 2, no Hamilton cycle exists.
    
    for v in range(n):
        if bin(adj[v]).count('1') < 2:
            return None
    
    # Backtracking DFS
    # To make it faster, order neighbors by some heuristic
    # Use recursion with pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    best_cycle = [None]
    found = [False]
    
    def dfs(current, visited, path):
        if found[0]:
            return
        if len(path) == n:
            # Check if current is connected to start
            if (adj[current] >> start) & 1:
                found[0] = True
                best_cycle[0] = path[:]
            return
        
        remaining = n - len(path)
        
        # Pruning: check connectivity of unvisited vertices + current
        # If unvisited vertices (plus current as potential bridge) are disconnected, prune
        unvisited_mask = ((1 << n) - 1) & ~visited
        if unvisited_mask:
            # Check if unvisited vertices form a connected subgraph (considering only edges among them and to current)
            # Actually we need: from current, can we reach all unvisited?
            # BFS/DFS from current through unvisited + current
            # But this is expensive. Simpler: check that no unvisited vertex is isolated from the rest.
            
            # Check: for each unvisited vertex, it must have at least one neighbor in (visited ∪ {current}) or in unvisited
            # Actually it needs to be reachable. Let's do a quick connectivity check.
            
            # Quick prune: if there's an unvisited vertex with no neighbors in unvisited and no neighbor is current, prune
            for u in range(n):
                if (unvisited_mask >> u) & 1:
                    # u is unvisited; it needs to be reachable
                    # It can connect to current or to other unvisited vertices
                    neighbors_in_unvisited = adj[u] & unvisited_mask
                    neighbor_is_current = (adj[u] >> current) & 1
                    if not neighbors_in_unvisited and not neighbor_is_current:
                        return  # u can never be reached
        
        # Try each unvisited neighbor of current
        neighbors = adj[current] & ~visited
        # Order: prefer vertices with fewer remaining options (most constrained first)
        # Count unvisited neighbors for each candidate
        candidates = []
        while neighbors:
            low = neighbors & (-neighbors)
            v = low.bit_length() - 1
            neighbors ^= low
            # Count unvisited neighbors of v
            unv_neighbors = bin(adj[v] & ~visited).count('1')
            candidates.append((unv_neighbors, v))
        candidates.sort()
        
        for _, v in candidates:
            path.append(v)
            visited |= (1 << v)
            dfs(v, visited, path)
            if found[0]:
                return
            path.pop()
            visited ^= (1 << v)
    
    dfs(start, start_mask, [start])
    
    if found[0]:
        return best_cycle[0]
    return None


def score_cycle(cycle):
    """Compute the max 1-based edge index for a cycle (list of N vertices, closing edge implicit)."""
    max_idx = 0
    # Build edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        edge_idx[(min(u,v), max(u,v))] = i
    
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        key = (min(u, v), max(u, v))
        idx = edge_idx[key]
        if idx > max_idx:
            max_idx = idx
    return max_idx


def construct():
    t_start = time.time()
    deadline = t_start + 80  # 80s deadline for safety
    
    # Binary search on K
    # Lower bound: N (need at least N edges for a cycle of length N, but actually need at least N edges)
    # Actually the minimum possible max index is at least N since we need N edges.
    # But edges are shared... the cycle has exactly N edges. The max index is at least the N-th smallest index among cycle edges.
    # Lower bound: N (1-based, need N distinct edges)
    # Upper bound: M (all edges)
    
    lo = N  # at least N edges needed
    hi = M  # all edges
    
    # First check if hi works
    adj_hi = build_adj_masks(hi)
    cycle_hi = has_hamilton_cycle(adj_hi)
    if cycle_hi is None:
        # Shouldn't happen for complete graph, but fallback
        return list(range(N))
    
    best_cycle = cycle_hi
    best_score = score_cycle(best_cycle)
    
    # Binary search
    while lo < hi and (time.time() - t_start) < deadline:
        mid = (lo + hi) // 2
        adj_mid = build_adj_masks(mid)
        cycle_mid = has_hamilton_cycle(adj_mid)
        if cycle_mid is not None:
            sc = score_cycle(cycle_mid)
            if sc < best_score:
                best_score = sc
                best_cycle = cycle_mid
            hi = mid
        else:
            lo = mid + 1
    
    # Try to improve with 2-opt-like local search on the found cycle
    # But the score is based on edge indices, not distances. 
    # We can try swapping to reduce the max edge index.
    
    if (time.time() - t_start) < deadline - 10:
        # Local improvement: try to reduce the max edge index
        edge_idx = {}
        for i, (u, v) in enumerate(EDGE_ORDER, 1):
            edge_idx[(min(u,v), max(u,v))] = i
        
        def get_cycle_edges(cyc):
            edges = []
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                edges.append((min(u,v), max(u,v)))
            return edges
        
        def cycle_max(cyc):
            mx = 0
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                key = (min(u,v), max(u,v))
                idx = edge_idx[key]
                if idx > mx:
                    mx = idx
            return mx
        
        improved = True
        while improved and (time.time() - t_start) < deadline - 5:
            improved = False
            cur = best_cycle[:]
            cur_score = cycle_max(cur)
            
            # Try 2-opt: reverse a segment
            for i in range(len(cur) - 1):
                if (time.time() - t_start) > deadline - 5:
                    break
                for j in range(i + 2, len(cur)):
                    # Reverse cur[i+1..j]
                    # Check if new edges are valid (exist in edge_idx) and improve
                    # New edges: (cur[i], cur[j]) and (cur[(j+1)%n], cur[i+1])
                    n = len(cur)
                    e1 = (min(cur[i], cur[j]), max(cur[i], cur[j]))
                    e2 = (min(cur[(j+1) % n], cur[i+1]), max(cur[(j+1) % n], cur[i+1]))
                    
                    if e1 not in edge_idx or e2 not in edge_idx:
                        continue
                    
                    # Compute new max
                    # Old edges removed: (cur[i], cur[i+1]) and (cur[j], cur[(j+1)%n])
                    # New edges added: e1 and e2
                    # Just compute new cycle max
                    new_cur = cur[:i+1] + cur[i+1:j+1][::-1] + cur[j+1:]
                    ns = cycle_max(new_cur)
                    if ns < cur_score:
                        cur = new_cur
                        cur_score = ns
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                best_cycle = cur
                best_score = cur_score
    
    return best_cycle


# Precompute edge index for faster access
_edge_idx_map = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_idx_map[(min(_u, _v), max(_u, _v))] = _i

# Override score_cycle to use precomputed map
def score_cycle(cycle):
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        key = (min(u, v), max(u, v))
        idx = _edge_idx_map[key]
        if idx > max_idx:
            max_idx = idx
    return max_idx
