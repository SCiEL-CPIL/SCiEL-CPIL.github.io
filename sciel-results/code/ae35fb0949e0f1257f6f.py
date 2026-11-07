import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 15
    # Partition: A = 0..6, B = 7..14
    # Base: K_{7,8}
    
    def build_base():
        edges = []
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u in range(7):
            for v in range(7, 15):
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return edges, adj, deg

    def can_add(u, v, adj, deg):
        # Check if adding (u,v) to graph with adj/deg is valid
        # New degrees: d'(u)=d(u)+1, d'(v)=d(v)+1, others same.
        
        # Case 1: s-u-v-t (middle edge)
        # s in N(u)\{v}, t in N(v)\{u,s}
        # Check d(s) != d(t)
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # Case 2: u-v-x-y (first edge)
        # x in N(v)\{u}, y in N(x)\{u,v}
        # Check d'(u) != d'(y) => d(u)+1 != d(y)
        du_new = deg[u] + 1
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[y] == du_new:
                    return False
        
        # Case 3: x-y-v-u (last edge)
        # x in N(u)\{v}, y in N(x)\{u,v}
        # Check d'(x) != d'(u) => d(x) != d(u)+1
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[x] == du_new:
                    return False
            
        return True

    def add_edge(u, v, edges, adj, deg):
        edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    # Generate intra-part candidates
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
            
    best_edges = []
    best_count = 0
    
    # Multiple randomized restarts
    num_restarts = 1000 # Adjust based on speed
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        base_edges, adj, deg = build_base()
        current_edges = list(base_edges)
        current_count = len(current_edges)
        
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        for u, v in shuffled_candidates:
            if time.time() > deadline:
                break
            if can_add(u, v, adj, deg):
                add_edge(u, v, current_edges, adj, deg)
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges[:]
            
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
