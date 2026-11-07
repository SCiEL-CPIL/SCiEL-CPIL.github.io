"""Triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_all_pairs(n, adj):
    """Compute all-pairs shortest paths. Returns dist matrix and diameter."""
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[src][u]
            for v in range(n):
                if adj[u][v] and dist[src][v] == -1:
                    dist[src][v] = du + 1
                    q.append(v)
    diam = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return None, -1  # disconnected
            if dist[i][j] > diam:
                diam = dist[i][j]
    return dist, diam

def _is_valid(n, added_edges, base_adj):
    """Full validation: no duplicates, triangle-free, connected, diameter <= 4."""
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        if u == v or adj[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        adj[u][v] = True
        adj[v][u] = True
    dist, diam = _bfs_all_pairs(n, adj)
    if dist is None:
        return False
    return diam <= 4

def _can_add_edge(adj, u, v, n):
    """Check if edge (u,v) can be added without creating a triangle."""
    if u == v or adj[u][v]:
        return False
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = P['n']
    base_edges = P['edges']
    base_adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        base_adj[u][v] = True
        base_adj[v][u] = True
    
    # Generate all candidate edges (cross-parity only, not in base)
    # In a path, same-parity edges always create triangles, so only cross-parity is valid
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v] and (u % 2) != (v % 2):
                candidates.append((u, v))
    
    # Precompute: for each candidate edge, which pairs does it help?
    # A pair (i,j) needs help if dist_path(i,j) > 4
    # Adding edge (u,v) helps pair (i,j) if it creates a shorter path
    
    # First, let's establish a known valid solution (8 edges from seed)
    seed_edges = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    if _is_valid(n, seed_edges, base_adj):
        best = [list(e) for e in seed_edges]
        best_count = 8
    else:
        best = []
        best_count = 99
    
    # Try to find solutions with fewer edges using structured search
    # Strategy: For k edges, try to find a valid set
    
    # Lower bound analysis:
    # In the path, endpoints 0 and 23 are 23 apart.
    # Each added edge (u,v) can reduce the distance by at most |u-v|-1 for pairs that route through it.
    # To get diameter 4, we need every pair within 4.
    # The "bottleneck" pairs are those far apart in the path.
    
    # Let's try k=5,6,7 with a smart search
    
    def _try_find_solution(k, time_limit=15.0):
        """Try to find a valid solution with exactly k edges."""
        t_start = time.time()
        t_end = t_start + time_limit
        random.seed(12345 + k)
        
        # Greedy initialization: pick edges that cover the most "hard" pairs
        # Hard pairs: those with path distance > 4
        hard_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if abs(i-j) > 4]
        
        def _edge_coverage(edge, current_edges):
            """How many hard pairs does this edge help, given current edges?"""
            # Build adjacency with current edges + this edge
            adj = [row[:] for row in base_adj]
            for u, v in current_edges:
                adj[u][v] = True
                adj[v][u] = True
            adj[edge[0]][edge[1]] = True
            adj[edge[1]][edge[0]] = True
            
            # Count how many hard pairs now have dist <= 4
            # But we only need to check if it helps at least one pair that was > 4 before
            # For speed, just check diameter
            dist, diam = _bfs_all_pairs(n, adj)
            if dist is None:
                return -1
            return -diam  # minimize diameter
        
        # Greedy: iteratively add the edge that most reduces diameter
        current = []
        adj = [row[:] for row in base_adj]
        
        for step in range(k):
            best_edge = None
            best_diam = 999
            for e in candidates:
                if not _can_add_edge(adj, e[0], e[1], n):
                    continue
                adj[e[0]][e[1]] = True
                adj[e[1]][e[0]] = True
                dist, diam = _bfs_all_pairs(n, adj)
                adj[e[0]][e[1]] = False
                adj[e[1]][e[0]] = False
                if diam < best_diam:
                    best_diam = diam
                    best_edge = e
            if best_edge is None:
                return None
            current.append(best_edge)
            adj[best_edge[0]][best_edge[1]] = True
            adj[best_edge[1]][best_edge[0]] = True
        
        # Check if valid
        if _is_valid(n, current, base_adj):
            return current
        
        # Local search: try swapping edges
        for iteration in range(2000):
            if time.time() > t_end:
                return None
            # Pick a random edge to remove
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            current[idx] = None  # mark as removed
            adj[removed[0]][removed[1]] = False
            adj[removed[1]][removed[0]] = False
            
            # Try to add a different edge
            best_add = None
            best_score = 999
            for e in candidates:
                if e == removed:
                    continue
                if not _can_add_edge(adj, e[0], e[1], n):
                    continue
                adj[e[0]][e[1]] = True
                adj[e[1]][e[0]] = True
                dist, diam = _bfs_all_pairs(n, adj)
                adj[e[0]][e[1]] = False
                adj[e[1]][e[0]] = False
                if dist is not None and diam < best_score:
                    best_score = diam
                    best_add = e
            
            if best_add is not None and best_score <= 4:
                current[idx] = best_add
                adj[best_add[0]][best_add[1]] = True
                adj[best_add[1]][best_add[0]] = True
                if _is_valid(n, current, base_adj):
                    return current
            else:
                # Revert
                current[idx] = removed
                adj[removed[0]][removed[1]] = True
                adj[removed[1]][removed[0]] = True
        
        return None
    
    # Try k = 5, 6, 7
    for k in [5, 6, 7]:
        if time.time() > deadline - 10:
            break
        result = _try_find_solution(k, time_limit=20.0)
        if result is not None:
            best = [list(e) for e in result]
            best_count = k
            break
    
    # Simulated annealing for further improvement
    if best_count > 5 and time.time() < deadline - 5:
        random.seed(777)
        current = [list(e) for e in best]
        current_set = set(map(tuple, current))
        
        def _fitness(edges):
            """Lower is better. Valid: -len(edges). Invalid: large penalty."""
            if _is_valid(n, edges, base_adj):
                return -len(edges)
            # Penalty based on how bad it is
            adj = [row[:] for row in base_adj]
            for u, v in edges:
                if u == v or adj[u][v]:
                    return 1000
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        return 1000
                adj[u][v] = True
                adj[v][u] = True
            dist, diam = _bfs_all_pairs(n, adj)
            if dist is None:
                return 1000
            return 100 + diam * 10 + len(edges)
        
        cur_fit = _fitness(current)
        best_fit = cur_fit
        best_sol = [list(e) for e in current]
        
        temp = 50.0
        cooling = 0.995
        iterations = 0
        max_iter = 50000
        
        while iterations < max_iter and time.time() < deadline - 2:
            iterations += 1
            if iterations % 5000 == 0:
                temp *= cooling
            
            # Random move: swap, add, or remove an edge
            move_type = random.random()
            new_edges = [list(e) for e in current]
            
            if move_type < 0.5 and len(new_edges) > 3:
                # Remove a random edge
                idx = random.randint(0, len(new_edges) - 1)
                new_edges.pop(idx)
            elif move_type < 0.8 and len(new_edges) < 12:
                # Add a random valid edge
                for attempt in range(20):
                    e = random.choice(candidates)
                    if e not in new_edges and (e[1], e[0]) not in new_edges:
                        new_edges.append(list(e))
                        break
            else:
                # Swap
                if len(new_edges) > 3:
                    idx = random.randint(0, len(new_edges) - 1)
                    for attempt in range(20):
                        e = random.choice(candidates)
                        if e not in new_edges and (e[1], e[0]) not in new_edges:
                            new_edges[idx] = list(e)
                            break
        
            new_fit = _fitness(new_edges)
            
            if new_fit <= best_fit:
                best_fit = new_fit
                best_sol = [list(e) for e in new_edges]
                current = new_edges
                cur_fit = new_fit
            elif random.random() < (2.71828 ** (-(new_fit - cur_fit) / max(temp, 0.1))):
                current = new_edges
                cur_fit = new_fit
        
        if best_fit < _fitness(best):
            best = best_sol
            best_count = len(best)
    
    # Final validation and type conversion
    if best and _is_valid(n, best, base_adj):
        return [[int(u), int(v)] for u, v in best]
    else:
        # Fallback to seed
        return [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]

# EVOLVE-BLOCK-END
