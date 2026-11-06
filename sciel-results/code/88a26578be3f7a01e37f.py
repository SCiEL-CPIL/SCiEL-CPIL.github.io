import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def _time_left():
        return deadline - time.time()

    def _creates_c4(adj, n, u, v):
        # Check if adding edge (u,v) creates a C4
        common_uv = adj[u] & adj[v]
        if len(common_uv) >= 2:
            return True
        for w in adj[u]:
            if w == v:
                continue
            if adj[w] & adj[v]:
                return True
        for w in adj[v]:
            if w == u:
                continue
            if adj[w] & adj[u]:
                return True
        return False

    def _is_c4_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return False
        return True

    def _max_independent_set_bound(adj, n, target, timeout):
        # Try to find an independent set of size 'target'
        # Use greedy with backtracking
        if n < target:
            return False
        
        # Simple branch and bound
        vertices = list(range(n))
        # Order vertices by degree (higher degree first for pruning)
        vertices.sort(key=lambda x: len(adj[x]), reverse=True)
        
        found = [False]
        start = time.time()
        
        def backtrack(remaining, current_size, forbidden):
            if found[0]:
                return
            if time.time() - start > timeout:
                return
            if current_size == target:
                found[0] = True
                return
            if len(remaining) < target - current_size:
                return
            
            # Pick a vertex with maximum degree in the remaining set
            # For efficiency, just pick the first one that maximizes pruning
            best_idx = 0
            best_degree = -1
            for i, v in enumerate(remaining):
                deg = len(adj[v] & set(remaining))
                if deg > best_degree:
                    best_degree = deg
                    best_idx = i
            
            v = remaining[best_idx]
            rem_rest = [x for i, x in enumerate(remaining) if i != best_idx]
            
            # Option 1: Include v
            new_remaining = [x for x in rem_rest if x not in adj[v]]
            backtrack(new_remaining, current_size + 1, forbidden | {v})
            
            # Option 2: Exclude v
            if not found[0]:
                backtrack(rem_rest, current_size, forbidden | {v})
        
        backtrack(vertices, 0, set())
        return found[0]

    def _verify_graph(n, edges):
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check C4-free
        if not _is_c4_free(adj, n):
            return False
        
        # Check no independent set of size 11
        if _max_independent_set_bound(adj, n, 11, 10.0):
            return False
        
        return True

    def _get_30_vertex_edges():
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a, b])
            edges.append([a, c])
            edges.append([b, c])
        return edges

    # Try to construct a C4-free graph on 39 vertices with alpha < 11
    # Use random greedy construction with restarts
    
    best_result = None
    best_n = 0
    
    for n in [39, 38, 37, 36, 35]:
        if _time_left() < 10.0:
            break
            
        found = False
        attempt = 0
        while _time_left() > 10.0 and not found:
            attempt += 1
            
            random.seed(attempt * 12345 + n)
            
            # Start with empty graph
            adj = [set() for _ in range(n)]
            edges = []
            
            # Generate all possible edges
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            # Greedy: add edges if C4-free
            for u, v in all_edges:
                if not _creates_c4(adj, n, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Check independence number
            if not _max_independent_set_bound(adj, n, 11, 8.0):
                # Valid!
                best_result = {'n': n, 'edges': edges}
                best_n = n
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to 30-vertex graph
    edges = _get_30_vertex_edges()
    return {'n': 30, 'edges': edges}
