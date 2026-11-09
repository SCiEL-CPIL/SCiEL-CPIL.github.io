import time
import sys
from itertools import combinations

# Precompute edge positions
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map: (min(u,v), max(u,v)) -> 1-based position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def cycle_score(cycle):
    """Compute the max edge position for a Hamilton cycle."""
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        p = get_pos(u, v)
        if p > mx:
            mx = p
    return mx

def find_cycle_fast(adj, n, deadline=None):
    """Find a Hamilton cycle using backtracking with strong pruning.
    adj: list of sets of neighbors.
    Returns list of vertices or None.
    """
    # Quick checks
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        u = stack.pop()
        count += 1
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                stack.append(w)
    if count != n:
        return None
    
    # Convert to sorted lists for ordering
    adj_list = [list(adj[v]) for v in range(n)]
    adj_set = [set(adj[v]) for v in range(n)]
    
    path = [0]
    used = [False] * n
    used[0] = True
    steps = [0]  # mutable counter for deadline checks
    
    def backtrack():
        if deadline is not None and time.time() > deadline:
            return None
        steps[0] += 1
        
        if len(path) == n:
            last = path[-1]
            if 0 in adj_set[last]:
                return path[:]
            return None
        
        last = path[-1]
        remaining = n - len(path)
        
        # Order neighbors: prefer those with fewer available options (Warnsdorff)
        # Also consider edge position for tie-breaking
        candidates = []
        for nxt in adj_list[last]:
            if not used[nxt]:
                # Count available neighbors for nxt (excluding used)
                avail = 0
                for nb in adj_set[nxt]:
                    if not used[nb]:
                        avail += 1
                # Special case: if this is the last step, nxt must connect to 0
                if remaining == 1:
                    if 0 not in adj_set[nxt]:
                        continue
                    # Also check that the one remaining unvisited vertex connects to both
                    unvisited = [v for v in range(n) if not used[v]]
                    # Actually nxt IS the last unvisited, so we just need nxt-0 edge
                    # and last-nxt edge (already checked)
                    candidates.append((0, get_pos(last, nxt), nxt))
                else:
                    # Pruning: if nxt has only 1 available neighbor (besides last), 
                    # it must be used next (forced)
                    if avail <= 1:
                        # This vertex will be a dead end unless it's the last one
                        # Actually if avail < 2 and remaining > 1, it's problematic
                        # But we need to be careful: avail counts unvisited neighbors
                        # If avail == 1, the only option is to go to that neighbor next
                        # If avail == 0, dead end (can't happen since last is used)
                        pass
                    candidates.append((avail, get_pos(last, nxt), nxt))
        
        if not candidates:
            return None
        
        candidates.sort()
        
        for _, _, nxt in candidates:
            if deadline is not None and time.time() > deadline:
                return None
            
            used[nxt] = True
            path.append(nxt)
            
            if remaining == 1:
                # Last vertex: must connect back to 0
                if 0 in adj_set[nxt]:
                    result = path[:]
                    path.pop()
                    used[nxt] = False
                    return result
                path.pop()
                used[nxt] = False
                continue
            
            # Additional pruning: check for isolated unvisited vertices
            # A vertex with 0 unvisited neighbors (and not last in path) is a dead end
            # But this is expensive, so only do it occasionally
            if steps[0] % 100 == 0:
                dead = False
                for v in range(n):
                    if not used[v] and v != nxt:
                        has_unvisited_neighbor = False
                        for nb in adj_set[v]:
                            if not used[nb]:
                                has_unvisited_neighbor = True
                                break
                        if not has_unvisited_neighbor:
                            dead = True
                            break
                if dead:
                    path.pop()
                    used[nxt] = False
                    continue
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            used[nxt] = False
        
        return None
    
    return backtrack()

def local_search_refine(cycle, n, deadline=None):
    """Try to improve a Hamilton cycle by swapping vertices to reduce max edge position."""
    if deadline is not None and time.time() > deadline:
        return cycle
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    # Try 2-opt style moves: swap two non-adjacent vertices
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # adjacent in cycle
                # Skip if i and j are adjacent
                if abs(i - j) == 1 or (i == 0 and j == n - 1):
                    continue
                
                # Create new cycle by swapping
                new_cycle = best_cycle[:]
                new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
                
                # Check if still a valid Hamilton cycle (it always is, since it's a permutation)
                new_score = cycle_score(new_cycle)
                
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    improved = True
                    if deadline is not None and time.time() > deadline:
                        return best_cycle
    
    # Try rotating segments
    for i in range(n):
        for j in range(i + 2, n):
            if deadline is not None and time.time() > deadline:
                return best_cycle
            # Reverse segment i..j
            new_cycle = best_cycle[:]
            seg = new_cycle[i:j+1]
            seg.reverse()
            new_cycle[i:j+1] = seg
            new_score = cycle_score(new_cycle)
            if new_score < best_score:
                best_score = new_score
                best_cycle = new_cycle
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    n = N
    total_edges = n * (n - 1) // 2  # 276
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Binary search for minimum threshold T where a Hamilton cycle exists
    lo, hi = 24, total_edges
    
    # First, find an upper bound by checking from a high threshold
    # Try progressively lower thresholds
    
    # Quick check: can we find any cycle at all?
    adj_full = [set() for _ in range(n)]
    for (u, v) in EDGE_ORDER:
        adj_full[u].add(v)
        adj_full[v].add(u)
    
    test_cycle = find_cycle_fast(adj_full, n, deadline)
    if test_cycle is not None:
        best_cycle = test_cycle
        best_score = cycle_score(best_cycle)
        hi = best_score  # We know a cycle exists with max edge pos = best_score
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [set() for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].add(v)
                adj[v].add(u)
        
        # Check for Hamilton cycle
        cycle = find_cycle_fast(adj, n, deadline)
        
        if cycle is not None:
            score = cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a cycle, try to refine it
    if best_cycle is not None:
        if time.time() < deadline:
            refined = local_search_refine(best_cycle, n, deadline)
            if refined is not None:
                rs = cycle_score(refined)
                if rs < best_score:
                    best_score = rs
                    best_cycle = refined
    
    # If no cycle found (shouldn't happen), fallback
    if best_cycle is None:
        # Try identity as last resort
        best_cycle = list(range(n))
        best_score = cycle_score(best_cycle)
    
    return best_cycle
