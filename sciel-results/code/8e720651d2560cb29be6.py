import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def verify_graph_fast(n, edges, k, g):
        """Fast verification using adjacency matrix and matrix powers for cycle detection."""
        if n == 0 or len(edges) != n * k // 2:
            return False
        
        # Build adjacency structure
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Fast girth check using non-backtracking walks
        # For girth g, we need to check cycles of length < g
        # Use BFS from each vertex with depth limit (g-1)//2
        max_depth = (g - 1) // 2
        
        for u in range(n):
            # BFS with non-backtracking constraint
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([(u, -1)])
            
            while q:
                curr, par = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append((v, curr))
                    elif v != parent[curr]:
                        # Found a cycle
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_random_k_regular_fast(n, k):
        """Faster random k-regular graph generation using configuration model with rejection."""
        if n * k % 2 != 0:
            return None
        
        # Use numpy for faster shuffling
        stubs = np.repeat(np.arange(n), k)
        np.random.shuffle(stubs)
        
        edges = []
        edge_set = set()
        valid = True
        
        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i+1])
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append((u, v))
        
        if not valid:
            return None
        return edges

    def luw_construct(q_val, d_val):
        """Lazebnik-Ustimenko-Woldar construction for (q^d)-regular graphs."""
        if q_val == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q_val
            sub = lambda x,y: (x-y)%q_val
            mul = lambda x,y: x*y%q_val
        
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        def get_rule(dv):
            r = dict(rule)
            for c in range(5, dv+1):
                rc = (c-5)%4
                c0 = c-(c-5)%4
                if rc==0: r[c]=(1,c0-2)
                elif rc==1: r[c]=(c0-1,1)
                elif rc==2: r[c]=(c0,1)
                else: r[c]=(1,c0+1)
            return r
        
        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0]*(d_val-1)
                for c in range(2, d_val+1):
                    a,b = r[c]
                    if side==0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d_val)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # Strategy:
    # 1. Try LUW construction first (deterministic, known to work for certain parameters)
    # 2. Then try random search from lower bound upward
    
    best_n = float('inf')
    best_edges = None
    
    # Try LUW with different d values
    # For k=4, g=10, we want a 4-regular graph with girth >= 10
    # LUW gives (q^d)-regular graphs, so for q=4, d=1 gives 4-regular
    # But d=1 is too small for girth 10. Try d=2, 3, etc.
    
    # Actually, LUW with q=4, d gives 4^d-regular, not 4-regular.
    # We need a different approach for 4-regular graphs.
    
    # For 4-regular graphs of girth 10, known constructions exist.
    # The (4,10)-cage is known to have 320 vertices.
    
    # Let's try a structured construction based on known cage graphs.
    # One approach: use the known 320-vertex (4,10)-cage if we can reconstruct it.
    
    # For now, let's try random search from n=320 upward.
    # But first, let's verify if n=320 is feasible.
    
    # Moore bound for (4,10): N >= 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161? 
    # Actually for even girth g=2m: N >= 1 + k*sum_{i=0}^{m-1}(k-1)^i
    # m=5, so sum = 1+3+9+27+81 = 121, N >= 1+4*121 = 485.
    # But the prompt says 320 is known, so there must be a better bound or my formula is wrong.
    # Let's trust the prompt: try from n=320.
    
    start_n = 320
    max_n = 600  # Don't go too high
    
    # Try random graphs
    for n in range(start_n, max_n + 1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random instances
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            edges = generate_random_k_regular_fast(n, k)
            if edges is None:
                continue
            
            if verify_graph_fast(n, edges, k, g):
                best_n = n
                best_edges = edges
                break
        
        if best_edges is not None:
            break
    
    # If random search failed, try LUW as fallback (even if it gives larger n)
    if best_edges is None:
        try:
            n_luw, edges_luw = luw_construct(k, 3)
            if verify_graph_fast(n_luw, edges_luw, k, g):
                best_n = n_luw
                best_edges = edges_luw
        except Exception:
            pass
    
    if best_edges is None:
        return 0, []
    
    return best_n, best_edges
