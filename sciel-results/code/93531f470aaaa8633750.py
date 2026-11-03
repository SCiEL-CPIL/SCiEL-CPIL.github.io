import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # LUW Construction - guaranteed valid graph
    def luw_construct(k, g):
        q = k
        # Find smallest d such that girth >= g
        # For odd d: girth >= d + 5
        # For even d: girth >= d + 4
        d = None
        for candidate_d in range(3, 20):
            if candidate_d % 2 == 1:  # odd
                if candidate_d + 5 >= g:
                    d = candidate_d
                    break
            else:  # even
                if candidate_d + 4 >= g:
                    d = candidate_d
                    break
        if d is None:
            d = g - 4
        if d < 3:
            d = 3

        if q == 4:
            _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def f_add(a, b): return a ^ b
            def f_sub(a, b): return a ^ b
            def f_mul(a, b): return _mul_t[a][b]
        else:
            def f_add(a, b): return (a + b) % q
            def f_sub(a, b): return (a - b) % q
            def f_mul(a, b): return (a * b) % q

        rules = {}
        rules[2] = (1, 1)
        rules[3] = (2, 1)
        rules[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            base = c - r
            candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
            rules[c] = candidates[r]

        start_key = (0, (0,) * (d + 1))
        label = {start_key: 0}
        queue = [start_key]
        edge_list = []

        for vkey in queue:
            side, coords = vkey
            vidx = label[vkey]
            for first_val in range(q):
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                    else:
                        ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
                nkey = (1 - side, tuple(ncoords))
                if nkey not in label:
                    label[nkey] = len(queue)
                    queue.append(nkey)
                if side == 0:
                    edge_list.append((vidx, label[nkey]))

        return len(queue), edge_list

    # Fast check: does the graph have any cycle shorter than g?
    # BFS from each vertex up to depth (g-1)//2, looking for back edges
    def has_short_cycle(adj, n, g, time_limit=None):
        """Returns True if there's a cycle of length < g. Uses BFS limited to depth."""
        max_depth = (g - 1) // 2  # We only need to check up to this depth
        # For cycle of length c < g, we need c <= g-1
        # In BFS from s, a cycle of length c is found when we see a visited node
        # at depth d where d + depth_of_visited + 1 = c
        # We only care about c < g, so we limit BFS depth
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
            if time_limit and time.time() > time_limit:
                return None  # Timeout, unknown
        
        return False  # No short cycle found

    def calculate_girth_fast(adj, n, g):
        """Calculate exact girth, but optimized to stop early if girth >= g."""
        # First check if girth >= g using the fast method
        result = has_short_cycle(adj, n, g)
        if result is False:
            return g  # Girth is at least g (could be larger, but we only need >= g)
        # If there IS a short cycle, we need the actual girth for SA
        # Do full BFS but with early termination
        girth = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                            if girth == 3:
                                return 3
            if girth == 3:
                return 3
        return girth if girth != float('inf') else n

    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        if n < k:
            return None
        if k > n - 1:
            return None
        # Configuration model with retries
        for attempt in range(50):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                # Verify regularity
                for i in range(n):
                    if len(adj[i]) != k:
                        valid = False
                        break
                if valid:
                    return adj
        return None

    def swap_edges(adj, n, k):
        # Build edge list efficiently
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if len(edges) < 2:
            return False
        
        idx1 = random.randint(0, len(edges) - 1)
        idx2 = random.randint(0, len(edges) - 1)
        attempts = 0
        while idx2 == idx1 and attempts < 10:
            idx2 = random.randint(0, len(edges) - 1)
            attempts += 1
        if idx2 == idx1:
            return False
        
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]
        
        options = [
            ((u1, u2), (v1, v2)),
            ((u1, v2), (v1, u2))
        ]
        random.shuffle(options)
        
        for e1, e2 in options:
            a1, b1 = e1
            a2, b2 = e2
            if a1 == b1 or a2 == b2:
                continue
            # Check new edges don't already exist
            if b1 in adj[a1] or b2 in adj[a2]:
                continue
            if a1 in adj[b1] or a2 in adj[b2]:
                continue
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            
            return True
        return False

    # Get LUW as baseline
    n_luw, edges_luw = luw_construct(k, g)
    
    # Moore Bound
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * k * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    # Ensure n*k is even
    n_start = moore_bound
    if k % 2 == 1 and n_start % 2 != 0:
        n_start += 1
    
    # Search for smaller n than LUW
    best_n = n_luw
    best_edges = edges_luw
    
    # Determine search range
    # Try n values from moore_bound up to n_luw - 1
    max_search_n = min(n_luw - 1, n_start + 200)  # Limit search range
    
    for n in range(n_start, max_search_n + 1):
        if time.time() > deadline - 10.0:
            break
        if k % 2 == 1 and n % 2 != 0:
            continue
        
        # Try multiple random restarts with SA
        found = False
        num_restarts = 8
        max_steps_per_restart = 3000
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            
            # Quick check: does it already have girth >= g?
            short = has_short_cycle(adj, n, g)
            if short is False:
                # Found valid graph!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n = n
                best_edges = edges
                found = True
                break
            elif short is None:
                # Timeout on girth check, skip this restart
                continue
            
            # SA to improve girth
            current_girth = calculate_girth_fast(adj, n, g)
            if current_girth >= g:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n = n
                best_edges = edges
                found = True
                break
            
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            steps = 0
            
            for step in range(max_steps_per_restart):
                if time.time() > deadline - 5.0:
                    break
                
                if not swap_edges(adj, n, k):
                    continue
                
                # Fast girth check
                short = has_short_cycle(adj, n, g)
                if short is False:
                    # Girth >= g, found valid graph!
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                elif short is None:
                    break
                
                # If still has short cycle, compute actual girth for SA
                new_girth = calculate_girth_fast(adj, n, g)
                delta = new_girth - current_girth
                if delta > 0 or random.random() < math.exp(delta / max(T, 0.01)):
                    current_girth = new_girth
                T *= cooling
                if T < min_T:
                    break
            
            if found:
                break
        
        if found:
            break
    
    if best_n < n_luw:
        return best_n, best_edges
    
    # Return LUW as fallback
    return n_luw, edges_luw
