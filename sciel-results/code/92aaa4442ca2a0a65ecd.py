"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 90.0  # seconds (evaluator kills at 90s, use safety margin)

    n = P['n']
    
    # Build initial adjacency from path
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True

    def is_triangle_free(adj, u, v):
        """Check if adding edge (u,v) creates a triangle."""
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    def bfs_all_pairs(adj):
        """Compute all-pairs shortest paths using BFS from each node.
        Returns 2D list dist[src][dst]."""
        dist = [[-1] * n for _ in range(n)]
        for src in range(n):
            dist[src][src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[src][v] == -1:
                        dist[src][v] = dist[src][u] + 1
                        queue.append(v)
        return dist

    def get_diameter(dist):
        """Get diameter from all-pairs distance matrix."""
        max_diam = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_diam:
                    max_diam = dist[i][j]
        return max_diam

    def check_valid(added_edges):
        """Verify the full set of added edges is valid."""
        # Check for duplicates and existing edges
        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if adj[a][b]:  # existing edge
                return False

        # Check triangle-free
        for u, v in added_edges:
            if not is_triangle_free(adj, u, v):
                return False

        # Check diameter
        tmp_adj = [row[:] for row in adj]
        for u, v in added_edges:
            tmp_adj[u][v] = True
            tmp_adj[v][u] = True
        dist = bfs_all_pairs(tmp_adj)
        return get_diameter(dist) <= 4

    # Set Cover greedy: select edges based on global pair coverage
    # A pair (i,j) is "covered" if dist[i][j] <= 4 in the augmented graph
    # We want to minimize edges such that all pairs are covered (diameter <= 4)
    
    # Precompute which pairs need coverage (all pairs with path distance > 4)
    path_dist = bfs_all_pairs(adj)
    
    # Pairs that need to be brought within distance 4
    uncovered_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            if path_dist[i][j] > 4:
                uncovered_pairs.add((i, j))
    
    if not uncovered_pairs:
        return []

    # Generate candidate edges (non-existing, triangle-free with current adj)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v] and is_triangle_free(adj, u, v):
                candidates.append((u, v))
    
    # For each candidate edge, compute which uncovered pairs it would cover
    # A pair (i,j) is covered by edge (u,v) if adding (u,v) makes dist(i,j) <= 4
    # We approximate: edge (u,v) covers pair (i,j) if the shortest path through
    # (u,v) is <= 4, i.e., min(dist_path(i,u)+1+dist_path(v,j), dist_path(i,v)+1+dist_path(u,j)) <= 4
    
    def edge_covers_pairs(u, v):
        """Return set of uncovered pairs that this edge would help cover."""
        covered = set()
        # For each uncovered pair, check if path through (u,v) gives dist <= 4
        for (i, j) in uncovered_pairs:
            # Path i->u->v->j: dist(i,u) + 1 + dist(v,j)
            d1 = path_dist[i][u] + 1 + path_dist[v][j]
            # Path i->v->u->j: dist(i,v) + 1 + dist(u,j)
            d2 = path_dist[i][v] + 1 + path_dist[u][j]
            if min(d1, d2) <= 4:
                covered.add((i, j))
        return covered
    
    # Precompute coverage for each candidate
    candidate_coverage = {}
    for (u, v) in candidates:
        candidate_coverage[(u, v)] = edge_covers_pairs(u, v)
    
    # Greedy set cover
    selected_edges = []
    current_uncovered = set(uncovered_pairs)
    current_adj = [row[:] for row in adj]
    
    # We need to maintain triangle-free constraint as we add edges
    # So we re-check triangle-free against the growing adjacency
    
    while current_uncovered and time.time() < deadline - 5:
        best_edge = None
        best_coverage = set()
        best_count = 0
        
        for (u, v) in candidates:
            # Check if edge is still valid (triangle-free with current adj)
            if current_adj[u][v]:
                continue
            if not is_triangle_free(current_adj, u, v):
                continue
            
            # Compute which currently uncovered pairs this edge covers
            # Use current distance approximation (path distances are lower bounds)
            covered = set()
            for (i, j) in current_uncovered:
                d1 = path_dist[i][u] + 1 + path_dist[v][j]
                d2 = path_dist[i][v] + 1 + path_dist[u][j]
                if min(d1, d2) <= 4:
                    covered.add((i, j))
            
            if len(covered) > best_count:
                best_count = len(covered)
                best_edge = (u, v)
                best_coverage = covered
        
        if best_edge is None or best_count == 0:
            break
        
        # Add the best edge
        u, v = best_edge
        current_adj[u][v] = True
        current_adj[v][u] = True
        selected_edges.append([u, v])
        current_uncovered -= best_coverage
    
    # Verify the solution
    if check_valid(selected_edges):
        return selected_edges
    
    # Fallback: try a different heuristic - prioritize edges that cover most pairs
    # with a penalty for edges that are "expensive" in terms of triangle-free constraint
    
    # Alternative: try specific known-good configurations
    # The star configuration from node 12 works with 8 edges
    fallback_edges = [
        [12, 1], [12, 3], [12, 6], [12, 9],
        [12, 15], [12, 18], [12, 21], [12, 23]
    ]
    if check_valid(fallback_edges):
        return fallback_edges
    
    # Try to reduce the fallback by removing redundant edges
    if selected_edges:
        # Try removing each edge and check if still valid
        reduced = list(selected_edges)
        for i in range(len(reduced)):
            test = reduced[:i] + reduced[i+1:]
            if check_valid(test):
                reduced = test
                break
        if check_valid(reduced):
            return reduced
    
    return selected_edges

# EVOLVE-BLOCK-END
