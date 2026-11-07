import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80

    # Build adjacency from base path
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)

    existing = set()
    for u, v in base_edges:
        existing.add((min(u,v), max(u,v)))

    # All possible candidate edges (non-path edges)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in existing:
                candidates.append((u, v))

    # Precompute which pairs need to be within distance 4
    # All pairs (i,j) with |i-j| > 4 need to be connected within 4 steps
    need_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if j - i > 4]

    # For each candidate edge, compute which need_pairs it helps cover
    # An edge (u,v) helps pair (i,j) if it can be part of a path of length <= 4 from i to j
    # More precisely: adding edge (u,v) creates new paths. A pair (i,j) is "covered"
    # if dist(i,j) <= 4 in the augmented graph.
    # For a single edge, the new shortest path from i to j using edge (u,v) is:
    #   min(dist_path(i,j), dist_path(i,u)+1+dist_path(v,j), dist_path(i,v)+1+dist_path(u,j))
    # where dist_path is the path distance (absolute difference on the line).
    # But we need to consider that the edge might be used in combination with others.
    
    # For set cover: an edge (u,v) "covers" pair (i,j) if adding just that edge
    # makes dist(i,j) <= 4. This is a necessary condition for the pair to be covered
    # by a set containing that edge, but not sufficient (other edges might help too).
    # Actually for set cover we want: a pair (i,j) is covered if there EXISTS a path
    # of length <= 4 in the augmented graph. This is complex.
    
    # Alternative: use the fact that for diameter <= 4, every pair must be within 4.
    # Let's use a different approach: BFS-based validation is fast enough.
    
    from collections import deque

    def check_diameter(adj2):
        """Check if diameter of graph with adjacency adj2 is <= 4."""
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj2[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
        return True

    def check_triangle_free(added):
        """Check if adding edges to base path keeps graph triangle-free."""
        adj2 = [set(adj[i]) for i in range(n)]
        for u, v in added:
            if len(adj2[u] & adj2[v]) > 0:
                return False
            adj2[u].add(v)
            adj2[v].add(u)
        return True

    def is_valid(added):
        if not check_triangle_free(added):
            return False
        adj2 = [set(adj[i]) for i in range(n)]
        for u, v in added:
            adj2[u].add(v)
            adj2[v].add(u)
        # Connected check
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj2[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        return check_diameter(adj2)

    # Known 7-edge solution as fallback
    hub_solution = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]
    best = hub_solution

    # Try to find solutions with fewer edges using systematic search
    # Key insight: with diameter 4 on 24 vertices, we need significant shortcuts.
    # Minimum edges: at least 2 (each edge can reduce diameter by at most ~11 in theory,
    # but triangle-free constraint limits this).
    
    # For k=2: try all pairs of candidates
    for k in range(2, 8):
        if time.time() > deadline:
            break
        
        if k == 2:
            # Try all pairs - there are ~276 candidates, so ~38k pairs
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(candidates)):
                    if time.time() > deadline:
                        break
                    added = [candidates[i], candidates[j]]
                    if is_valid(added):
                        best = added
                        return [[int(u), int(v)] for u, v in best]
        
        elif k == 3:
            # Try triples - too many to brute force all, use smart sampling
            # Focus on edges that are "far apart" to maximize coverage
            # Sample based on spread
            import random
            random.seed(42)
            # Generate diverse triples
            for _ in range(200000):
                if time.time() > deadline:
                    break
                # Pick 3 random candidates with some structure
                idx1 = random.randint(0, len(candidates)-1)
                idx2 = random.randint(0, len(candidates)-1)
                idx3 = random.randint(0, len(candidates)-1)
                if idx1 == idx2 or idx1 == idx3 or idx2 == idx3:
                    continue
                added = [candidates[idx1], candidates[idx2], candidates[idx3]]
                if is_valid(added):
                    best = added
                    return [[int(u), int(v)] for u, v in best]
        
        elif k == 4:
            import random
            random.seed(123)
            for _ in range(300000):
                if time.time() > deadline:
                    break
                idxs = random.sample(range(len(candidates)), 4)
                added = [candidates[i] for i in idxs]
                if is_valid(added):
                    best = added
                    return [[int(u), int(v)] for u, v in best]
        
        elif k == 5:
            import random
            random.seed(456)
            for _ in range(200000):
                if time.time() > deadline:
                    break
                idxs = random.sample(range(len(candidates)), 5)
                added = [candidates[i] for i in idxs]
                if is_valid(added):
                    best = added
                    return [[int(u), int(v)] for u, v in best]
        
        elif k == 6:
            import random
            random.seed(789)
            for _ in range(150000):
                if time.time() > deadline:
                    break
                idxs = random.sample(range(len(candidates)), 6)
                added = [candidates[i] for i in idxs]
                if is_valid(added):
                    best = added
                    return [[int(u), int(v)] for u, v in best]
        
        elif k == 7:
            # Already have a 7-edge solution, but try to find another
            break

    return [[int(u), int(v)] for u, v in best]
