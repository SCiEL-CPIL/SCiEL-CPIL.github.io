"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _bfs_distances(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    queue = [src]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in range(n):
            if adj[u][v] and dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def _all_pairs_diameter(n, adj):
    max_d = 0
    for src in range(n):
        dist = _bfs_distances(n, adj, src)
        for d in dist:
            if d == -1:
                return -1
            if d > max_d:
                max_d = d
    return max_d

def _is_triangle_free_add(adj, u, v, n):
    if adj[u][v]:
        return False
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return False
    return True

def _get_fair_pairs(n, adj, threshold=5):
    """Get all pairs with distance >= threshold (need to be reduced)."""
    far_pairs = set()
    for src in range(n):
        dist = _bfs_distances(n, adj, src)
        for dst in range(src + 1, n):
            if dist[dst] >= threshold:
                far_pairs.add((src, dst))
    return far_pairs

def _edge_covers_pairs(n, adj, u, v, far_pairs):
    """For a candidate edge (u,v), determine which far pairs it helps reduce.
    An edge (u,v) helps pair (a,b) if adding it reduces dist(a,b) below threshold."""
    if not _is_triangle_free_add(adj, u, v, n):
        return set()
    
    # Temporarily add edge
    adj[u][v] = True
    adj[v][u] = True
    
    # Recompute distances and check which far pairs are now < threshold
    covered = set()
    for a, b in far_pairs:
        # Check if dist(a,b) is now < threshold
        # Use BFS from a
        dist = _bfs_distances(n, adj, a)
        if dist[b] < 5:
            covered.add((a, b))
    
    # Remove edge
    adj[u][v] = False
    adj[v][u] = False
    
    return covered

def _greedy_set_cover(n, base_adj, far_pairs, valid_edges, deadline):
    """Greedy set cover: repeatedly pick the edge that covers the most uncovered pairs."""
    remaining = set(far_pairs)
    selected = []
    adj = [row[:] for row in base_adj]
    
    # Precompute coverage for all valid edges
    coverage = {}
    for u, v in valid_edges:
        if time.time() > deadline:
            break
        cov = _edge_covers_pairs(n, adj, u, v, remaining)
        if cov:
            coverage[(u, v)] = cov
    
    while remaining and time.time() < deadline:
        # Find edge covering most remaining pairs
        best_edge = None
        best_count = 0
        for edge, cov in coverage.items():
            if time.time() > deadline:
                break
            count = len(cov & remaining)
            if count > best_count:
                best_count = count
                best_edge = edge
        
        if best_edge is None or best_count == 0:
            break
        
        u, v = best_edge
        selected.append((u, v))
        remaining -= coverage[best_edge]
        
        # Update adj
        adj[u][v] = True
        adj[v][u] = True
        
        # Recompute remaining far pairs (some may now be covered by combination)
        new_far = _get_fair_pairs(n, adj, 5)
        remaining = new_far
        
        # Recompute coverage for remaining edges with updated adj
        for edge in list(coverage.keys()):
            if edge in [(s[0], s[1]) for s in selected]:
                continue
            cov = _edge_covers_pairs(n, adj, edge[0], edge[1], remaining)
            if cov:
                coverage[edge] = cov
            else:
                del coverage[edge]
    
    return selected

def _local_search_reduce(n, base_adj, initial_edges, deadline):
    """Try to reduce the number of edges by removing one and finding a replacement."""
    edges = list(initial_edges)
    
    for _ in range(10):  # Limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                break
            
            # Try removing edge i
            candidate = [e for j, e in enumerate(edges) if j != i]
            
            # Check if candidate is valid (connected, diameter <= 4, triangle-free)
            adj = [row[:] for row in base_adj]
            valid = True
            for u, v in candidate:
                if not _is_triangle_free_add(adj, u, v, n):
                    valid = False
                    break
                adj[u][v] = True
                adj[v][u] = True
            
            if not valid:
                continue
            
            diam = _all_pairs_diameter(n, adj)
            if diam != -1 and diam <= 4:
                edges = candidate
                improved = True
                break
        
        if not improved:
            break
    
    return edges

def _validate_solution(n, base_adj, added_edges):
    """Final validation: connected, triangle-free, diameter <= 4, no duplicates."""
    adj = [row[:] for row in base_adj]
    edge_set = set()
    for u, v in added_edges:
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        if not _is_triangle_free_add(adj, u, v, n):
            return False
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add(key)
    
    diam = _all_pairs_diameter(n, adj)
    return diam != -1 and diam <= 4

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])
    
    # Get all valid single edges (triangle-free with base)
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v] and _is_triangle_free_add(base_adj, u, v, n):
                valid_edges.append((u, v))
    
    # Get far pairs in base graph (distance >= 5)
    far_pairs = _get_fair_pairs(n, base_adj, 5)
    
    # Use greedy set cover to find minimal edge set
    solution = _greedy_set_cover(n, base_adj, far_pairs, valid_edges, deadline)
    
    # Validate the solution
    if not _validate_solution(n, base_adj, solution):
        # Fallback: use a known-good seed
        solution = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    
    # Try to reduce further with local search
    if time.time() < deadline:
        reduced = _local_search_reduce(n, base_adj, solution, deadline)
        if _validate_solution(n, base_adj, reduced):
            solution = reduced
    
    # Final validation
    if not _validate_solution(n, base_adj, solution):
        solution = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    
    return [list(e) for e in solution]

# EVOLVE-BLOCK-END
