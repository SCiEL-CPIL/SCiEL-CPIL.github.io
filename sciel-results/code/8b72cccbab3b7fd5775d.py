"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
    return True

def _bfs_diameter(adj, start):
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    queue = [start]
    idx = 0
    max_d = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                max_d = max(max_d, dist[v])
                queue.append(v)
    if any(d == -1 for d in dist):
        return -1
    return max_d

def _diameter(adj):
    n = len(adj)
    # Check connectivity from node 0
    d0 = _bfs_diameter(adj, 0)
    if d0 == -1:
        return -1
    # For diameter we need max over all pairs; use BFS from each node
    # But for small n this is fine. Optimize: only need to check if <= 4
    for s in range(n):
        d = _bfs_diameter(adj, s)
        if d == -1 or d > 4:
            return d if d != -1 else -1
    return d0

def _valid_solution(added_edges, base_edges, n):
    all_edges = list(base_edges) + added_edges
    # Check no duplicate edges
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    adj = _build_adj(n, all_edges)
    if not _is_triangle_free(adj):
        return False
    diam = _diameter(adj)
    if diam == -1 or diam > 4:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    n = P['n']
    base_edges = P['edges']
    
    # Initial solution: star from center 12 to all odd vertices except neighbors
    center = 12
    initial_edges = []
    for v in range(n):
        if v % 2 != 0 and abs(center - v) != 1:
            u, w = (center, v) if center < v else (v, center)
            initial_edges.append([u, w])
    
    best = initial_edges
    
    # Verify initial solution
    if _valid_solution(initial_edges, base_edges, n):
        best = initial_edges
    else:
        # Fallback: try a few other centers
        for c in range(n):
            edges = []
            for v in range(n):
                if v != c and abs(c - v) != 1 and (c % 2) != (v % 2):
                    u, w = (c, v) if c < v else (v, c)
                    edges.append([u, w])
            if _valid_solution(edges, base_edges, n):
                best = edges
                break
    
    # Branch and bound: try to find solutions with fewer edges
    # Strategy: use star structures from various centers
    # A star from center c to vertices in set S (same parity, not adjacent) 
    # has |S| edges. We need diameter <= 4.
    
    # For a path P_n, adding a star from center c to all vertices of one parity
    # (excluding path neighbors) gives a bipartite graph.
    # Diameter analysis: for center c, the farthest vertex is at distance 
    # max(c, n-1-c) along the path, but with the star, vertices on the same
    # side can reach via center in 2 hops.
    
    # Key insight: if we use center c and connect to ALL vertices of parity p 
    # (where p = 1 - c%2) that are not path-neighbors, then:
    # - Vertices of parity p reach center in 1 hop
    # - Vertices of parity 1-p reach center in 2 hops (via path neighbor)
    # - So diameter is at most 4 (parity p -> center -> parity 1-p -> center -> parity p)
    #   Actually: parity p to parity p: p->c->p' is 2 hops if both connected to c
    #   parity p to parity 1-p: p->c->(path neighbor)->p' is 3 hops
    #   Wait, let me reconsider.
    
    # Let's just search over subsets of the star edges.
    # For each center, the candidate edges are to same-parity non-adjacent vertices.
    # We want the minimum subset that gives diameter <= 4 and connected.
    
    for center in range(n):
        if time.time() > deadline:
            break
        
        # Candidate edges for this center: to vertices of opposite parity, not adjacent
        candidates = []
        for v in range(n):
            if v == center:
                continue
            if abs(center - v) == 1:
                continue  # already in path
            # For triangle-free with path: if center connects to v, and v is adjacent
            # to u in path, then center-u must NOT be an edge. Since path is bipartite,
            # center and v must have opposite parity for the star to remain bipartite.
            # Actually, the path is bipartite (even/odd). Adding edges from center to
            # same-parity vertices keeps it bipartite. Adding to opposite-parity would
            # create odd cycles (triangles) with path edges.
            # So we can only connect to SAME parity vertices.
            if (center % 2) == (v % 2):
                u, w = (center, v) if center < v else (v, center)
                candidates.append([u, w])
        
        if len(candidates) == 0:
            continue
        
        # Try all subsets of size k from 1 upward
        # But this is exponential. Use greedy + local search instead.
        
        # Greedy: start with empty, add edges that most reduce diameter
        # But first, check if the full star works
        if _valid_solution(candidates, base_edges, n):
            if len(candidates) < len(best):
                best = list(candidates)
        
        # Now try to remove edges from the full star
        # Try removing one edge at a time
        current = list(candidates)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                trial = current[:i] + current[i+1:]
                if _valid_solution(trial, base_edges, n):
                    current = trial
                    improved = True
                    break
            if time.time() > deadline:
                break
        
        if len(current) < len(best):
            best = current
    
    # Also try: two-center approach - split the path into two halves
    # Each half gets a star center
    if time.time() < deadline:
        for c1 in range(n // 4, 3 * n // 4):
            for c2 in range(c1 + 2, n):
                if time.time() > deadline:
                    break
                # c1 handles left portion, c2 handles right portion
                # Both must connect to same-parity vertices
                if (c1 % 2) != (c2 % 2):
                    continue
                edges = []
                for v in range(n):
                    if v == c1 or v == c2:
                        continue
                    if abs(c1 - v) == 1 or abs(c2 - v) == 1:
                        continue
                    if (c1 % 2) == (v % 2):
                        # Assign to closer center
                        d1 = abs(c1 - v)
                        d2 = abs(c2 - v)
                        if d1 <= d2:
                            u, w = (c1, v) if c1 < v else (v, c1)
                            edges.append([u, w])
                        else:
                            u, w = (c2, v) if c2 < v else (v, c2)
                            edges.append([u, w])
                
                if _valid_solution(edges, base_edges, n):
                    if len(edges) < len(best):
                        best = edges
    
    # Convert to proper types
    result = []
    for e in best:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
