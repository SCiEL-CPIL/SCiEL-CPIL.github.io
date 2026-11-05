"""Search for a Hamilton cycle with minimal maximum edge arrival index."""
import time
import sys

# The edge list from the pinned shuffle
EDGE_LIST = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24
TOTAL_EDGES = len(EDGE_LIST)

# Build edge index: edge_index[(u,v)] = 0-based index in EDGE_LIST
def _build_edge_index():
    idx = {}
    for i, (u, v) in enumerate(EDGE_LIST):
        if u > v:
            u, v = v, u
        idx[(u, v)] = i
    return idx

EDGE_INDEX = _build_edge_index()

def _get_edge_idx(u, v):
    """Get 0-based index of edge (u,v) in the shuffled list."""
    if u > v:
        u, v = v, u
    return EDGE_INDEX[(u, v)]

def _cycle_max_index(cycle):
    """Compute the max 0-based edge index in the Hamilton cycle.
    cycle is a list of N vertices in order; closing edge is (cycle[-1], cycle[0])."""
    mx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        e = _get_edge_idx(u, v)
        if e > mx:
            mx = e
    return mx

def _has_hamilton_cycle_in_prefix(k):
    """Check if there's a Hamilton cycle using only edges with index < k.
    Uses bitmask DFS with pruning. k is the number of edges available (0-based count).
    Returns True if a Hamilton cycle exists in G_k (first k edges)."""
    if k < N:
        return False
    
    # Build adjacency as bitmasks
    adj = [0] * N
    for i in range(k):
        u, v = EDGE_LIST[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check minimum degree
    for v in range(N):
        if bin(adj[v]).count('1') < 2:
            return False
    
    # Bitmask DFS with pruning
    # Fix vertex 0 as start to break rotational symmetry
    # Also, we can fix the direction by requiring the second vertex < the last vertex
    # But for simplicity, let's just fix vertex 0 as start.
    
    from functools import lru_cache
    
    # Use iterative DFS with backtracking and pruning
    # State: (current_vertex, visited_mask)
    # We want to find a path from 0 visiting all vertices and returning to 0.
    
    # For efficiency, use a recursive approach with memoization on (vertex, mask)
    # But the state space is 24 * 2^24 which is too large.
    # Instead, use backtracking with good ordering.
    
    # Better approach: use the fact that we can fix vertex 0 as the first vertex.
    # Then we need to find a path from some neighbor of 0 through all other vertices to another neighbor of 0.
    
    # Let's use a DFS that builds the path from vertex 0.
    # Pruning: 
    # 1. If the unvisited vertices form a disconnected component that can't be reached, prune.
    # 2. If the current vertex has no unvisited neighbors and we haven't visited all, prune.
    # 3. Connectivity check: the unvisited subgraph + current vertex must be connected to allow completion.
    
    # For N=24, a well-ordered DFS should work for dense enough graphs.
    
    path = [0]
    visited = 1  # bitmask
    
    def dfs(pos, curr, mask):
        if pos == N:
            # Check if curr can connect back to 0
            return (mask & (1 << 0)) != 0 and ((adj[curr] >> 0) & 1) != 0
        
        # Pruning: check if remaining unvisited vertices are reachable
        # Quick check: current vertex must have at least one unvisited neighbor (unless pos == N-1)
        remaining = ((1 << N) - 1) & ~mask
        if pos < N - 1:
            # Need at least one unvisited neighbor
            if (adj[curr] & remaining) == 0:
                return False
        
        # Connectivity pruning: check that unvisited vertices are connected to current
        # This is expensive, so only do it at certain depths
        if pos % 4 == 0 and pos > 0:
            # BFS from current through unvisited vertices
            seen = 1 << curr
            stack = [curr]
            while stack:
                v = stack.pop()
                neighbors = adj[v] & remaining
                while neighbors:
                    nb = (neighbors & -neighbors).bit_length() - 1
                    neighbors &= neighbors - 1
                    if not (seen >> nb) & 1:
                        seen |= (1 << nb)
                        stack.append(nb)
            # All unvisited vertices must be reachable
            if (seen & remaining) != remaining:
                return False
        
        # Try neighbors in order of increasing edge index (prefer early edges)
        # Get unvisited neighbors
        unvisited_neighbors = adj[curr] & remaining
        if unvisited_neighbors == 0 and pos < N - 1:
            return False
        
        # Order neighbors by how many unvisited neighbors they have (most constrained first)
        if unvisited_neighbors:
            nbs = []
            tmp = unvisited_neighbors
            while tmp:
                nb = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                # Count unvisited neighbors of nb
                cnt = bin(adj[nb] & remaining & ~(1 << nb)).count('1')
                nbs.append((cnt, nb))
            nbs.sort()
            for _, nb in nbs:
                if dfs(pos + 1, nb, mask | (1 << nb)):
                    return True
        elif pos == N - 1:
            # Last vertex, check connection back to 0
            return (adj[curr] >> 0) & 1
        
        return False
    
    return dfs(1, 0, 1)

def _find_hamilton_cycle_in_prefix(k):
    """Find a Hamilton cycle using only edges with index < k.
    Returns the cycle as a list of vertices, or None if not found."""
    if k < N:
        return None
    
    adj = [0] * N
    for i in range(k):
        u, v = EDGE_LIST[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check minimum degree
    for v in range(N):
        if bin(adj[v]).count('1') < 2:
            return None
    
    path = []
    
    def dfs(pos, curr, mask):
        path.append(curr)
        if pos == N:
            if (mask & (1 << 0)) != 0 and ((adj[curr] >> 0) & 1) != 0:
                return True
            path.pop()
            return False
        
        remaining = ((1 << N) - 1) & ~mask
        if pos < N - 1:
            if (adj[curr] & remaining) == 0:
                path.pop()
                return False
        
        if pos % 4 == 0 and pos > 0:
            seen = 1 << curr
            stack = [curr]
            while stack:
                v = stack.pop()
                neighbors = adj[v] & remaining
                while neighbors:
                    nb = (neighbors & -neighbors).bit_length() - 1
                    neighbors &= neighbors - 1
                    if not (seen >> nb) & 1:
                        seen |= (1 << nb)
                        stack.append(nb)
            if (seen & remaining) != remaining:
                path.pop()
                return False
        
        unvisited_neighbors = adj[curr] & remaining
        if unvisited_neighbors == 0 and pos < N - 1:
            path.pop()
            return False
        
        if unvisited_neighbors:
            nbs = []
            tmp = unvisited_neighbors
            while tmp:
                nb = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                cnt = bin(adj[nb] & remaining & ~(1 << nb)).count('1')
                nbs.append((cnt, nb))
            nbs.sort()
            for _, nb in nbs:
                if dfs(pos + 1, nb, mask | (1 << nb)):
                    return True
        elif pos == N - 1:
            if (adj[curr] >> 0) & 1:
                return True
        
        path.pop()
        return False
    
    if dfs(1, 0, 1):
        return list(path)
    return None

def _two_opt_improve(cycle, max_allowed):
    """Try to improve the cycle by 2-opt moves, keeping max edge index <= max_allowed."""
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                if j == i + 1:
                    continue
                # Consider reversing segment i+1..j
                a = cycle[i]
                b = cycle[i + 1]
                c = cycle[j]
                d = cycle[(j + 1) % N]
                
                # Check if edges (a,c) and (b,d) exist within max_allowed
                try:
                    e1 = _get_edge_idx(a, c)
                    e2 = _get_edge_idx(b, d)
                except KeyError:
                    continue
                
                if e1 > max_allowed or e2 > max_allowed:
                    continue
                
                # Check if this 2-opt actually reduces the max
                old_max = max(_get_edge_idx(a, b), _get_edge_idx(c, d))
                new_max = max(e1, e2)
                
                if new_max < old_max:
                    # Perform the 2-opt
                    cycle[i+1:j+1] = reversed(cycle[i+1:j+1])
                    improved = True
                    break
            if improved:
                break
    return cycle

def _local_search(cycle, max_allowed, time_limit):
    """Perform local search to minimize max edge index, starting from cycle."""
    start_time = time.time()
    best_cycle = list(cycle)
    best_max = _cycle_max_index(best_cycle)
    
    # Try 2-opt and 3-opt moves
    while time.time() - start_time < time_limit:
        improved = False
        # 2-opt
        for i in range(N - 1):
            for j in range(i + 2, N):
                a = best_cycle[i]
                b = best_cycle[i + 1]
                c = best_cycle[j]
                d = best_cycle[(j + 1) % N]
                
                try:
                    e1 = _get_edge_idx(a, c)
                    e2 = _get_edge_idx(b, d)
                except KeyError:
                    continue
                
                old_max = _cycle_max_index(best_cycle)
                # Compute new max after 2-opt
                new_cycle = list(best_cycle)
                new_cycle[i+1:j+1] = reversed(new_cycle[i+1:j+1])
                new_max = _cycle_max_index(new_cycle)
                
                if new_max < old_max:
                    best_cycle = new_cycle
                    best_max = new_max
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_cycle, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline to be safe
    
    # Binary search for the minimum k such that a Hamilton cycle exists in prefix k
    lo, hi = N, TOTAL_EDGES
    
    # First, find the minimal k using binary search
    while lo < hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        if _has_hamilton_cycle_in_prefix(mid):
            hi = mid
        else:
            lo = mid + 1
    
    min_k = lo
    
    # Now find an actual cycle at min_k
    cycle = _find_hamilton_cycle_in_prefix(min_k)
    
    if cycle is None:
        # Fallback: try larger k
        for k in range(min_k + 1, min(100, TOTAL_EDGES + 1)):
            if time.time() > deadline - 5:
                break
            cycle = _find_hamilton_cycle_in_prefix(k)
            if cycle is not None:
                break
    
    if cycle is None:
        # Last resort: return identity permutation
        return list(range(N))
    
    # Try to improve with local search
    if time.time() < deadline - 5:
        cycle, _ = _local_search(cycle, min_k, deadline - time.time() - 5)
    
    # The score is the largest one-based arrival index
    # Our cycle uses edges with 0-based indices < min_k (at most)
    # The score is max_index + 1 (one-based)
    return list(cycle)
