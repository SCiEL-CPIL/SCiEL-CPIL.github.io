"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 16}
DEADLINE = time.time() + 80  # safety margin before 90s kill

def _is_valid(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj[a].add(b)
        adj[b].add(a)
    
    # Check diameter exactly 2
    def bfs_dist(start):
        dist = {start: 0}
        queue = [start]
        for node in queue:
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist
    
    max_d = 0
    for i in range(n):
        d = bfs_dist(i)
        if len(d) < n:
            return False  # disconnected
        for j in range(n):
            if i != j:
                if d[j] > 2:
                    return False
                max_d = max(max_d, d[j])
    if max_d != 2:
        return False  # must be exactly 2, not 1
    
    # Check every edge is critical
    for (u, v) in edge_set:
        # Remove edge (u,v)
        adj2 = [set(s) for s in adj]
        adj2[u].discard(v)
        adj2[v].discard(u)
        
        # Check connectivity and diameter <= 2
        def bfs_dist2(start):
            dist = {start: 0}
            queue = [start]
            for node in queue:
                for nb in adj2[node]:
                    if nb not in dist:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            return dist
        
        ok = True
        for i in range(n):
            d = bfs_dist2(i)
            if len(d) < n:
                ok = False
                break
            for j in range(n):
                if i != j and d[j] > 2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False  # edge not critical
    
    return True

def _edge_count_validity_check(edges):
    """Quick check: all edges critical via common-neighbor rule (faster than full BFS for many cases)."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u, v in edges:
        # Edge is critical if it's a bridge OR N(u) ∩ N(v) = ∅
        # Quick check: if they share a common neighbor, edge might not be critical
        # But it could still be critical if it's a bridge. For safety, use full check in _is_valid.
        # This helper is just for fast rejection.
        common = adj[u] & adj[v]
        if common:
            # Might still be critical if bridge, but in dense graphs unlikely. 
            # We'll let _is_valid make the final call.
            pass
    return True

def construct():
    start_time = time.time()
    n = P['n']
    
    # Start with a star (always valid, 15 edges)
    best_edges = [[0, v] for v in range(1, n)]
    best_count = len(best_edges)
    
    # Try to build a denser graph
    # Strategy: start with a random graph, then iteratively remove non-critical edges
    # and try to add critical edges.
    
    # First, try a known good dense construction if we can find one quickly.
    # For now, do a local search from the star.
    
    random.seed(42)
    
    for iteration in range(200):
        if time.time() > DEADLINE:
            break
        
        # Try adding a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        
        # Check if edge already exists
        exists = False
        for e in best_edges:
            if (e[0], e[1]) == (a, b) or (e[0], e[1]) == (b, a):
                exists = True
                break
        if exists:
            continue
        
        new_edges = best_edges + [[a, b]]
        
        # Check if still valid
        if _is_valid(new_edges):
            best_edges = new_edges
            best_count = len(best_edges)
    
    # Ensure types are correct
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
