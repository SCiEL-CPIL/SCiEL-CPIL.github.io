"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0

    n = P['n']

    # Build initial adjacency from path
    base_adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u][v] = True
        base_adj[v][u] = True

    def bfs_diameter(adj):
        """Compute diameter using BFS from each node. Returns (diam, dist_matrix)."""
        max_diam = 0
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
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam, dist_matrix

    def is_triangle_free_with_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    def check_full_validity(added_edges):
        """Full validation: no dupes, no existing edges, triangle-free, connected, diam<=4."""
        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if base_adj[a][b]:
                return False

        adj = [row[:] for row in base_adj]
        for u, v in added_edges:
            adj[u][v] = True
            adj[v][u] = True

        # Check triangle-free
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False

        diam, _ = bfs_diameter(adj)
        return diam <= 4

    # Hardcoded known-good 8-edge solution as fallback
    hardcoded = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]

    if check_full_validity(hardcoded):
        best_solution = hardcoded
    else:
        best_solution = None

    # NEW STRUCTURAL APPROACH: Rooted eccentricity formulation
    # For a fixed root r, we need all vertices within distance <= 2 from r
    # in the "auxiliary" sense. Actually, for diameter <= 4, it's sufficient
    # to find a root r such that max distance from r is <= 2 (then any two
    # vertices are at most 4 apart via r). But we need triangle-freeness.
    #
    # Better: For diameter <= 4, pick a center vertex c. All vertices must
    # be within distance 2 of c. This is a dominating set problem with radius 2.
    # Each added edge (u,v) where u is adjacent to c helps cover v's neighbors.
    #
    # Even better formulation: 
    # - Pick a root r
    # - We need: for every vertex v, dist(r,v) <= 2
    # - dist(r,v) <= 1 if v is neighbor of r (base or added)
    # - dist(r,v) <= 2 if v is neighbor of some neighbor of r
    # - An added edge (r, x) makes x a neighbor of r, so x's neighbors become dist 2
    # - An added edge (a, b) where a is neighbor of r makes b potentially dist 2
    #   if b was previously far
    #
    # For a path, let's enumerate possible roots and solve the covering problem.

    def solve_rooted(root):
        """Find minimum edges to make all vertices within dist 2 of root, triangle-free."""
        adj = [row[:] for row in base_adj]
        
        # Initial distances from root
        def get_dist_from_root(adj, root):
            dist = [-1] * n
            dist[root] = 0
            queue = [root]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            return dist

        dist = get_dist_from_root(adj, root)
        
        # Vertices that need coverage: those with dist > 2
        def get_uncovered(dist):
            return [v for v in range(n) if dist[v] > 2]
        
        uncovered = get_uncovered(dist)
        if not uncovered:
            return []
        
        selected = []
        
        # Candidate edges that are triangle-free and could help
        # An edge (u,v) helps if it reduces some vertex's distance to root
        # Most useful: edges from root to far vertices, or from near-root to far vertices
        
        while uncovered and time.time() < deadline - 5:
            best_edge = None
            best_gain = 0
            
            # Generate candidates
            cands = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v] and is_triangle_free_with_edge(adj, u, v):
                        cands.append((u, v))
            
            # Score each candidate by how many uncovered vertices it covers
            for u, v in cands:
                # Simulate adding this edge
                adj[u][v] = True
                adj[v][u] = True
                new_dist = get_dist_from_root(adj, root)
                adj[u][v] = False
                adj[v][u] = False
                
                new_uncovered = [x for x in uncovered if new_dist[x] > 2]
                gain = len(uncovered) - len(new_uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (u, v)
            
            if best_edge is None or best_gain == 0:
                break
            
            u, v = best_edge
            selected.append([u, v])
            adj[u][v] = True
            adj[v][u] = True
            dist = get_dist_from_root(adj, root)
            uncovered = get_uncovered(dist)
        
        if not uncovered:
            return selected
        return None

    # Try all possible roots (focus on central ones for efficiency)
    # For a path of 24 vertices, roots 8-15 are most promising
    for root in range(8, 16):
        if time.time() > deadline - 10:
            break
        result = solve_rooted(root)
        if result is not None:
            # Verify full validity (diameter, not just rooted distance)
            if check_full_validity(result):
                if best_solution is None or len(result) < len(best_solution):
                    best_solution = result

    # Try two-root approach: pick two roots, ensure all vertices within dist 2 of at least one
    # This allows diameter <= 4 (via either root)
    def solve_two_rooted(root1, root2):
        """Find minimum edges so all vertices within dist 2 of root1 OR root2."""
        adj = [row[:] for row in base_adj]
        
        def get_min_dist(adj, r1, r2):
            """Get min distance to either root."""
            def bfs(r):
                dist = [-1] * n
                dist[r] = 0
                queue = [r]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    for v in range(n):
                        if adj[u][v] and dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                return dist
            d1 = bfs(r1)
            d2 = bfs(r2)
            return [min(d1[i], d2[i]) if d1[i] != -1 and d2[i] != -1 else (d1[i] if d1[i] != -1 else d2[i]) for i in range(n)]
        
        dist = get_min_dist(adj, root1, root2)
        
        def get_uncovered(dist):
            return [v for v in range(n) if dist[v] > 2]
        
        uncovered = get_uncovered(dist)
        if not uncovered:
            return []
        
        selected = []
        
        while uncovered and time.time() < deadline - 5:
            best_edge = None
            best_gain = 0
            
            cands = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v] and is_triangle_free_with_edge(adj, u, v):
                        cands.append((u, v))
            
            for u, v in cands:
                adj[u][v] = True
                adj[v][u] = True
                new_dist = get_min_dist(adj, root1, root2)
                adj[u][v] = False
                adj[v][u] = False
                
                new_uncovered = [x for x in uncovered if new_dist[x] > 2]
                gain = len(uncovered) - len(new_uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (u, v)
            
            if best_edge is None or best_gain == 0:
                break
            
            u, v = best_edge
            selected.append([u, v])
            adj[u][v] = True
            adj[v][u] = True
            dist = get_min_dist(adj, root1, root2)
            uncovered = get_uncovered(dist)
        
        if not uncovered:
            return selected
        return None

    # Try pairs of roots (central pairs)
    for r1 in range(6, 13):
        for r2 in range(r1 + 1, 18):
            if time.time() > deadline - 8:
                break
            result = solve_two_rooted(r1, r2)
            if result is not None:
                if check_full_validity(result):
                    if best_solution is None or len(result) < len(best_solution):
                        best_solution = result

    # Try to further improve by removing edges
    if best_solution:
        current = list(best_solution)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(len(current)):
                trial = current[:idx] + current[idx+1:]
                if check_full_validity(trial):
                    current = trial
                    improved = True
                    break
        if len(current) < len(best_solution):
            best_solution = current

    # Final fallback
    if best_solution is None:
        if check_full_validity(hardcoded):
            best_solution = hardcoded
        else:
            return []

    # Ensure correct types
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
