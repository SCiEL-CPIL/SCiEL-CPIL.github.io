"""Find a Hamilton cycle minimizing the max edge arrival index via binary search + backtracking."""
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

# Build edge index: for each (u,v) pair (u<v), store its arrival index (1-based)
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_idx[(a, b)] = i + 1

def get_edge_weight(u, v):
    a, b = min(u, v), max(u, v)
    return edge_idx.get((a, b), M + 1)

def cycle_score(cycle):
    """Compute the max arrival index over all edges in the Hamilton cycle."""
    n = len(cycle)
    max_arrival = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        w = get_edge_weight(u, v)
        if w > max_arrival:
            max_arrival = w
    return max_arrival

def build_adjacency(threshold):
    """Build adjacency list for edges with arrival index <= threshold."""
    adj = [set() for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER):
        if i + 1 > threshold:
            break
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_hamilton_cycle(adj, deadline):
    """Find a Hamilton cycle in the graph defined by adj using backtracking with pruning."""
    # Use bitmask DP with backtracking for n=24
    # We'll use a recursive backtracking approach with degree pruning
    
    # Check minimum degree - every vertex must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Sort vertices by degree for ordering
    order = sorted(range(N), key=lambda x: len(adj[x]))
    
    # Try to find a Hamilton cycle starting from the vertex with minimum degree
    start = order[0]
    
    # Use iterative deepening with backtracking
    # State: current vertex, set of visited vertices (as bitmask), path so far
    # For n=24, we use a recursive approach with memoization on (current, visited_mask)
    
    # Actually for n=24, bitmask has 2^24 = 16M states which is too many.
    # Use backtracking with good heuristics instead.
    
    sys.setrecursionlimit(10000)
    
    def backtrack(current, visited_mask, path):
        if time.time() > deadline:
            return None
        if len(path) == N:
            # Check if we can close the cycle
            if start in adj[current]:
                return list(path)
            return None
        
        # Pruning: check if any unvisited vertex has lost all connections
        # (i.e., all its neighbors in the remaining graph are either visited 
        #  and not adjacent to current, or it's isolated)
        remaining = N - len(path)
        
        # Try neighbors in order of fewest remaining connections
        neighbors = list(adj[current])
        # Filter to unvisited
        unvisited_neighbors = [v for v in neighbors if not (visited_mask >> v) & 1]
        
        if not unvisited_neighbors:
            return None
        
        # Sort by degree (prefer vertices with fewer options - most constrained first)
        unvisited_neighbors.sort(key=lambda v: len(adj[v]))
        
        for next_v in unvisited_neighbors:
            new_mask = visited_mask | (1 << next_v)
            new_path = path + [next_v]
            
            # Pruning: check if any unvisited vertex (not next_v, not current) 
            # has no unvisited neighbors and isn't adjacent to current (for closing)
            # and isn't the last vertex to be placed
            if remaining > 1:
                # Check connectivity of unvisited vertices
                # A simpler check: for each unvisited vertex w (not next_v),
                # it must have at least one neighbor that is either:
                #   - unvisited (will be visited later), or
                #   - is current (if it's the last one to be placed)
                # Actually simpler: just check that no unvisited vertex is completely isolated
                # from the remaining unvisited set + {current}
                isolated = False
                for w in range(N):
                    if w == current or w == next_v:
                        continue
                    if (visited_mask >> w) & 1:
                        continue  # already visited
                    # w is unvisited, check if it has any connection to remaining
                    # Its neighbors must include at least one unvisited vertex or current
                    # (since current is the last visited, it can only connect to the final vertex)
                    has_connection = False
                    for nb in adj[w]:
                        if nb == current:
                            # w can connect to current only if w is the last vertex
                            # We don't know that yet, so allow it
                            has_connection = True
                            break
                        if not (new_mask >> nb) & 1:
                            # nb is still unvisited after placing next_v
                            has_connection = True
                            break
                    if not has_connection:
                        isolated = True
                        break
                if isolated:
                    continue
            
            result = backtrack(next_v, new_mask, new_path)
            if result is not None:
                return result
        
        return None
    
    initial_mask = 1 << start
    result = backtrack(start, initial_mask, [start])
    return result

def find_hamilton_cycle_fast(adj, deadline):
    """Faster Hamilton cycle finder using Warnsdorff's rule heuristic."""
    sys.setrecursionlimit(10000)
    
    # Check minimum degree
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Try multiple starting vertices
    # Use the vertex with minimum degree as start
    start_candidates = sorted(range(N), key=lambda x: len(adj[x]))
    
    for start in start_candidates[:5]:  # Try up to 5 starting vertices
        def backtrack(current, visited_mask, path):
            if time.time() > deadline:
                return None
            if len(path) == N:
                if start in adj[current]:
                    return list(path)
                return None
            
            remaining = N - len(path)
            
            # Get unvisited neighbors
            neighbors = [v for v in adj[current] if not (visited_mask >> v) & 1]
            
            if not neighbors:
                return None
            
            # Warnsdorff's rule: prefer vertices with fewer onward connections
            def onward_count(v):
                count = 0
                for nb in adj[v]:
                    if not (visited_mask >> nb) & 1 and nb != current:
                        count += 1
                # If v is the last to be placed, it needs to connect to start
                if remaining == 1:
                    if start in adj[v]:
                        count = 0  # Most preferred
                return count
            
            neighbors.sort(key=onward_count)
            
            for next_v in neighbors:
                new_mask = visited_mask | (1 << next_v)
                new_path = path + [next_v]
                
                # Quick pruning: check if any unvisited vertex is isolated
                if remaining > 1:
                    isolated = False
                    for w in range(N):
                        if (visited_mask >> w) & 1 or w == next_v:
                            continue
                        # w is unvisited
                        # Check if w has any neighbor that is unvisited or is current
                        has_conn = False
                        for nb in adj[w]:
                            if nb == current:
                                has_conn = True
                                break
                            if not (new_mask >> nb) & 1:
                                has_conn = True
                                break
                        if not has_conn:
                            isolated = True
                            break
                    if isolated:
                        continue
                
                result = backtrack(next_v, new_mask, new_path)
                if result is not None:
                    return result
            
            return None
        
        initial_mask = 1 << start
        result = backtrack(start, initial_mask, [start])
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Binary search on the threshold
    # Low: minimum possible (N, since we need at least N edges in a cycle)
    # High: M (all edges)
    lo = N  # A cycle needs N edges, so threshold must be at least N
    hi = M
    
    # First, find a feasible upper bound by trying progressively larger thresholds
    # Start from the middle and adjust
    best_cycle = None
    best_score = M + 1
    
    # Try to find the minimum threshold
    # Use binary search
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        adj = build_adjacency(mid)
        
        cycle = find_hamilton_cycle_fast(adj, deadline)
        
        if cycle is not None:
            # Found a cycle with all edges <= mid
            score = cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search didn't find anything (shouldn't happen since full graph has cycles)
    if best_cycle is None:
        # Fallback: use greedy
        best_cycle = list(range(N))
        best_score = cycle_score(best_cycle)
    
    # Verify and return
    if best_cycle is not None and len(best_cycle) == N:
        return best_cycle
    
    # Ultimate fallback
    return list(range(N))

# EVOLVE-BLOCK-END
