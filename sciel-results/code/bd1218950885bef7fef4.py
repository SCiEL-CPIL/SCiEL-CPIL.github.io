"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 82.0

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(base_edges, added_edges, n):
    """Build adjacency masks from base and added edges."""
    adj = [0] * n
    edge_set = set()
    for u, v in base_edges:
        u, v = min(u, v), max(u, v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
    for u, v in added_edges:
        u, v = min(u, v), max(u, v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
    return adj, edge_set

def _bfs_dist(adj, n, start):
    """BFS from start, return list of distances."""
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        m = adj[u]
        while m:
            lb = m & -m
            v = lb.bit_length() - 1
            m ^= lb
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_valid(n, base_edges, added_edges):
    """Check: no duplicates, triangle-free, connected, diameter <= 4."""
    adj, edge_set = _build_adj(base_edges, added_edges, n)
    
    # Check for duplicate edges (added edges shouldn't be in base)
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    for u, v in added_edges:
        u, v = min(u, v), max(u, v)
        if (u, v) in base_set:
            return False
    
    # Triangle-free check
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                if (adj[i] & adj[j]) != 0:
                    return False
    
    # Connected + diameter check
    for s in range(n):
        dist = _bfs_dist(adj, n, s)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _get_diameter(adj, n):
    """Compute diameter of graph."""
    max_d = 0
    for s in range(n):
        dist = _bfs_dist(adj, n, s)
        for d in dist:
            if d > max_d:
                max_d = d
    return max_d

def _get_pairs_exceeding(adj, n, threshold=4):
    """Get all pairs with distance > threshold."""
    pairs = []
    for s in range(n):
        dist = _bfs_dist(adj, n, s)
        for t in range(s + 1, n):
            if dist[t] > threshold:
                pairs.append((s, t))
    return pairs

def _greedy_set_cover(n, base_edges):
    """
    Greedy set cover approach:
    - Universe: all vertex pairs with path distance > 4
    - Each candidate edge covers pairs that become distance <= 4
    """
    adj, edge_set = _build_adj(base_edges, [], n)
    base_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    # Get all candidate edges (triangle-free compatible)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            if (adj[i] & adj[j]) == 0:
                candidates.append((i, j))
    
    # Initial pairs to cover
    uncovered = set(_get_pairs_exceeding(adj, n))
    
    added = []
    while uncovered and _time_ok():
        best_edge = None
        best_coverage = 0
        
        for u, v in candidates:
            # Check if still triangle-free with current adj
            if (adj[u] & adj[v]) != 0:
                continue
            
            # Temporarily add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Count how many uncovered pairs this covers
            coverage = 0
            for (a, b) in uncovered:
                # Check if distance between a and b is now <= 4
                # Quick check: if a==u or a==v or b==u or b==v, use new edge
                # Otherwise, need to recompute or use heuristic
                # For efficiency, just check if the edge helps
                if (a == u and b == v) or (a == v and b == u):
                    coverage += 1
                    continue
                # Check if going through u-v helps
                dist_a_to_u = -1
                dist_a_to_v = -1
                dist_b_to_u = -1
                dist_b_to_v = -1
                
                # Simple BFS from a to check if b is reachable within 4
                # This is expensive, so use a heuristic:
                # If a and b are both within distance 2 of {u,v}, they might be covered
                pass
            
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Use a simpler coverage metric: count pairs that include u or v
            # and are currently uncovered
            for (a, b) in uncovered:
                if a == u or a == v or b == u or b == v:
                    coverage += 1
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_edge = (u, v)
        
        if best_edge is None or best_coverage == 0:
            break
        
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        added.append([u, v])
        
        # Update uncovered pairs
        uncovered = set(_get_pairs_exceeding(adj, n))
    
    return added

def _local_search(n, base_edges, added, max_iterations=100):
    """Local search: try removing edges, then swapping."""
    if not added:
        return added
    
    # Try removing edges
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            trial = added[:i] + added[i + 1:]
            if _check_valid(n, base_edges, trial):
                added = trial
                changed = True
    
    # Try swaps
    for _ in range(max_iterations):
        if not _time_ok():
            break
        improved = False
        adj, edge_set = _build_adj(base_edges, added, n)
        base_set = set((min(u, v), max(u, v)) for u, v in base_edges)
        
        # Find candidate edges
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in edge_set:
                    continue
                if i % 2 == j % 2:
                    continue
                if (adj[i] & adj[j]) == 0:
                    candidates.append((i, j))
        
        for i in range(len(added)):
            if not _time_ok():
                break
            removed = added[i]
            for cand in candidates:
                if not _time_ok():
                    break
                if cand == removed or cand == (removed[1], removed[0]):
                    continue
                trial = added[:i] + added[i + 1:] + [list(cand)]
                if _check_valid(n, base_edges, trial):
                    added = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return added

def _branch_and_bound(n, base_edges, upper_bound):
    """
    Branch and bound for minimum edge set.
    This is a heuristic B&B that tries to find a solution with fewer edges.
    """
    adj, edge_set = _build_adj(base_edges, [], n)
    base_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    # Get all candidate edges
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            if (adj[i] & adj[j]) == 0:
                candidates.append((i, j))
    
    # Sort candidates by some heuristic (e.g., degree)
    def edge_key(e):
        u, v = e
        return (adj[u].bit_count(), adj[v].bit_count())
    candidates.sort(key=edge_key, reverse=True)
    
    best_solution = None
    best_count = upper_bound
    
    def search(current_edges, idx, depth):
        nonlocal best_solution, best_count
        
        if not _time_ok():
            return
        
        if len(current_edges) >= best_count:
            return
        
        # Check if current solution is valid
        if _check_valid(n, base_edges, current_edges):
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_solution = current_edges[:]
            return
        
        # If we've exhausted candidates, check validity
        if idx >= len(candidates):
            return
        
        # Get pairs still exceeding diameter
        adj_temp, _ = _build_adj(base_edges, current_edges, n)
        pairs_exceeding = _get_pairs_exceeding(adj_temp, n)
        
        if not pairs_exceeding:
            return
        
        # Choose a pair to cover
        a, b = pairs_exceeding[0]
        
        # Option 1: skip this candidate
        search(current_edges, idx + 1, depth + 1)
        
        # Option 2: take this candidate if it helps
        u, v = candidates[idx]
        if (adj_temp[u] & adj_temp[v]) == 0:
            new_edges = current_edges + [[u, v]]
            adj_temp[u] |= (1 << v)
            adj_temp[v] |= (1 << u)
            search(new_edges, idx + 1, depth + 1)
    
    # Start search with a greedy initial solution
    greedy_added = _greedy_set_cover(n, base_edges)
    if greedy_added and len(greedy_added) < best_count:
        best_count = len(greedy_added)
        best_solution = greedy_added
    
    # Run B&B with limited depth
    search([], 0, 0)
    
    return best_solution if best_solution else greedy_added

def construct():
    n = P['n']
    base = P['edges']
    
    # Start with greedy set cover
    added = _greedy_set_cover(n, base)
    
    # Verify
    if not _check_valid(n, base, added):
        # Fallback: try dense bipartite + prune
        adj, _ = _build_adj(base, [], n)
        base_set = set((min(u, v), max(u, v)) for u, v in base)
        
        dense = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in base_set:
                    continue
                if i % 2 == j % 2:
                    continue
                dense.append([i, j])
        
        # Prune
        changed = True
        while changed and _time_ok():
            changed = False
            for i in range(len(dense) - 1, -1, -1):
                if not _time_ok():
                    break
                trial = dense[:i] + dense[i + 1:]
                if _check_valid(n, base, trial):
                    dense = trial
                    changed = True
        
        if _check_valid(n, base, dense):
            added = dense
        else:
            return []
    
    # Local search to reduce
    added = _local_search(n, base, added)
    
    # Final verification
    if _check_valid(n, base, added):
        return [[int(u), int(v)] for u, v in added]
    
    return []

# EVOLVE-BLOCK-END
