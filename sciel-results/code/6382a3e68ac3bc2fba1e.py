"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds

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
        """Compute all-pairs shortest paths using BFS from each node."""
        dist_matrix = []
        for src in range(n):
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
            dist_matrix.append(dist)
        return dist_matrix

    def get_diameter(dist_matrix):
        """Get diameter from distance matrix."""
        max_diam = 0
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][j] > max_diam:
                    max_diam = dist_matrix[i][j]
        return max_diam

    def get_pairs_needing_reduction(dist_matrix, target_diam):
        """Get pairs where distance > target_diam."""
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i][j] > target_diam:
                    pairs.append((i, j))
        return pairs

    # Hardcoded solution from the seed program (8 edges)
    added_edges = [
        [12, 1], [12, 3], [12, 6], [12, 9],
        [12, 15], [12, 18], [12, 21], [12, 23]
    ]

    # Verify hardcoded solution
    def check_valid(added_edges):
        """Verify the full set of added edges is valid."""
        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if adj[a][b]:
                return False

        tmp_adj = [row[:] for row in adj]
        for u, v in added_edges:
            tmp_adj[u][v] = True
            tmp_adj[v][u] = True
        
        # Check triangle-free
        for u in range(n):
            for v in range(u+1, n):
                if tmp_adj[u][v]:
                    for w in range(n):
                        if w != u and w != v and tmp_adj[u][w] and tmp_adj[v][w]:
                            return False
        
        dist_matrix = bfs_all_pairs(tmp_adj)
        return get_diameter(dist_matrix) <= 4

    if check_valid(added_edges):
        return added_edges

    # Set Cover Greedy: select edges based on global pair coverage
    current_adj = [row[:] for row in adj]
    current_dist = bfs_all_pairs(current_adj)
    current_diam = get_diameter(current_dist)
    
    if current_diam <= 4:
        return []

    # Generate all candidate edges (non-existing, triangle-free)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if not current_adj[u][v] and is_triangle_free(current_adj, u, v):
                candidates.append((u, v))

    # For each candidate, compute which "far" pairs it would help reduce
    # A pair (i,j) with dist > 4 needs to be reduced to <= 4
    # Adding edge (u,v) helps pair (i,j) if it creates a shorter path
    
    def edge_covers_pairs(u, v, dist_matrix, target_diam):
        """Determine which pairs with dist > target_diam would be reduced by adding edge (u,v)."""
        covered = []
        # Compute new distances considering the added edge
        # For each pair (i,j), new_dist = min(old_dist, dist(i,u)+1+dist(v,j), dist(i,v)+1+dist(u,j))
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i][j] <= target_diam:
                    continue
                # Check if adding edge (u,v) reduces this pair's distance
                d1 = dist_matrix[i][u] + 1 + dist_matrix[v][j] if dist_matrix[i][u] != -1 and dist_matrix[v][j] != -1 else float('inf')
                d2 = dist_matrix[i][v] + 1 + dist_matrix[u][j] if dist_matrix[i][v] != -1 and dist_matrix[u][j] != -1 else float('inf')
                new_dist = min(dist_matrix[i][j], d1, d2)
                if new_dist <= target_diam:
                    covered.append((i, j))
        return covered

    # Greedy set cover: repeatedly pick edge that covers most uncovered pairs
    uncovered_pairs = set(get_pairs_needing_reduction(current_dist, 4))
    selected_edges = []
    
    while uncovered_pairs and time.time() < deadline - 5:
        best_edge = None
        best_coverage = 0
        best_covered_pairs = set()
        
        for u, v in candidates:
            if (u, v) in selected_edges or (v, u) in selected_edges:
                continue
            # Check if still triangle-free with current selected edges
            # (simplified: we check against current_adj which includes selected edges)
            if not is_triangle_free(current_adj, u, v):
                continue
            
            covered = edge_covers_pairs(u, v, current_dist, 4)
            # Only count pairs that are currently uncovered
            new_coverage = len([p for p in covered if p in uncovered_pairs])
            
            if new_coverage > best_coverage:
                best_coverage = new_coverage
                best_edge = (u, v)
                best_covered_pairs = set(covered)
        
        if best_edge is None or best_coverage == 0:
            break
        
        u, v = best_edge
        selected_edges.append([u, v])
        current_adj[u][v] = True
        current_adj[v][u] = True
        current_dist = bfs_all_pairs(current_adj)
        uncovered_pairs = set(get_pairs_needing_reduction(current_dist, 4))
    
    # Verify the solution
    if uncovered_pairs:
        # Fallback to hardcoded if greedy failed
        if check_valid(added_edges):
            return added_edges
        return []
    
    return selected_edges

# EVOLVE-BLOCK-END
