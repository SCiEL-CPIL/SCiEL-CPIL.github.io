import random
import time

def construct():
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def check_edge_valid(u, v, adj, deg):
        """
        Checks if edge (u,v) is valid in the graph defined by adj and deg.
        Note: adj and deg must represent the graph *without* (u,v) if we are testing addition,
        or *with* (u,v) if we are testing existence.
        Here, we assume (u,v) is NOT in adj/deg yet, and we want to know if we CAN add it.
        The condition for (u,v) to be valid in the NEW graph:
        The degrees of neighbors of u (excluding v) in the NEW graph must be disjoint from
        the degrees of neighbors of v (excluding u) in the NEW graph.
        
        In the NEW graph:
        - deg_new(u) = deg(u) + 1
        - deg_new(v) = deg(v) + 1
        - For any x in N(u) \ {v}, deg_new(x) = deg(x) (unchanged)
        - For any y in N(v) \ {u}, deg_new(y) = deg(y) (unchanged)
        
        So we check:
        S_u = { deg(x) for x in adj[u] if x != v }
        S_v = { deg(y) for y in adj[v] if y != u }
        Return S_u.isdisjoint(S_v)
        """
        degs_u = set(deg[x] for x in adj[u] if x != v)
        degs_v = set(deg[y] for y in adj[v] if y != u)
        return degs_u.isdisjoint(degs_v)

    def check_edge_valid_existing(u, v, adj, deg):
        """
        Checks if edge (u,v) is valid in the graph defined by adj and deg,
        ASSUMING (u,v) IS IN adj and deg reflects the graph WITH (u,v).
        """
        # Neighbors of u excluding v
        # Their degrees are deg[x]
        # Neighbors of v excluding u
        # Their degrees are deg[y]
        degs_u = set(deg[x] for x in adj[u] if x != v)
        degs_v = set(deg[y] for y in adj[v] if y != u)
        return degs_u.isdisjoint(degs_v)

    def greedy_construct(seed):
        random.seed(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        candidates = all_edges[:]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in edges or (v, u) in edges:
                continue
            
            # 1. Check if (u,v) itself would be valid if added
            if not check_edge_valid(u, v, adj, deg):
                continue
            
            # 2. Check if adding (u,v) invalidates any existing edge incident to u or v
            # An existing edge (u, w) is affected because N(u) gains v.
            # The new degree of v is deg[v] + 1.
            # We need to check if (u, w) remains valid.
            # Condition for (u, w): 
            # S_u_new = (S_u_old \ {w}) U {deg_new(v)}  ... wait.
            # Let's just simulate the addition and check the specific edges.
            
            valid_addition = True
            
            # Check (u, v) is valid (already done)
            
            # Check all neighbors w of u (excluding v, which is not there yet)
            for w in list(adj[u]):
                if w == v: continue
                # The edge is (u, w).
                # In the new graph, N(u) includes v.
                # The set of degrees for N(u)\{w} will include deg_new(v) = deg[v] + 1.
                # The set of degrees for N(w)\{u} remains unchanged (since v is not neighbor of w, 
                # unless v==w which is impossible, or v is already neighbor of w? 
                # If v is neighbor of w, then (v,w) is an edge. Adding (u,v) doesn't change deg(w) 
                # unless w is u or v. So deg(w) is unchanged.
                # So S_w = {deg(x) for x in adj[w] if x != u} is unchanged.
                
                # We need to check if S_u_new intersects S_w.
                # S_u_new = {deg(x) for x in adj[u] if x != w} U {deg[v] + 1}
                # Note: v is not in adj[u] yet.
                
                S_u = set(deg[x] for x in adj[u] if x != w)
                S_u.add(deg[v] + 1)
                
                S_w = set(deg[y] for y in adj[w] if y != u)
                
                if S_u & S_w:
                    valid_addition = False
                    break
            
            if not valid_addition:
                continue
            
            # Check all neighbors x of v (excluding u)
            for x in list(adj[v]):
                if x == u: continue
                # The edge is (v, x).
                # In the new graph, N(v) includes u.
                # The set of degrees for N(v)\{x} will include deg_new(u) = deg[u] + 1.
                
                S_v = set(deg[y] for y in adj[v] if y != x)
                S_v.add(deg[u] + 1)
                
                S_x = set(deg[z] for z in adj[x] if z != v)
                
                if S_v & S_x:
                    valid_addition = False
                    break
            
            if valid_addition:
                # Commit addition
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        
        return len(edges), list(edges)

    def local_search_add(edges_list, seed):
        """
        Try to add more edges to a valid graph.
        """
        random.seed(seed + 1000)
        edges = set(edges_list)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
        improved = True
        while improved:
            improved = False
            candidates = [e for e in all_edges if e not in edges]
            random.shuffle(candidates)
            for u, v in candidates:
                # Same check as greedy_construct
                if not check_edge_valid(u, v, adj, deg):
                    continue
                
                valid_addition = True
                for w in list(adj[u]):
                    if w == v: continue
                    S_u = set(deg[x] for x in adj[u] if x != w)
                    S_u.add(deg[v] + 1)
                    S_w = set(deg[y] for y in adj[w] if y != u)
                    if S_u & S_w:
                        valid_addition = False
                        break
                if not valid_addition:
                    continue
                for x in list(adj[v]):
                    if x == u: continue
                    S_v = set(deg[y] for y in adj[v] if y != x)
                    S_v.add(deg[u] + 1)
                    S_x = set(deg[z] for z in adj[x] if z != v)
                    if S_v & S_x:
                        valid_addition = False
                        break
                if valid_addition:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    improved = True
        
        return len(edges), list(edges)

    start_time = time.time()
    max_time = 55
    best_count = 0
    best_edges = []
    
    for i in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Greedy construction
        count, edges = greedy_construct(i)
        
        # Local search to fill gaps
        count2, edges2 = local_search_add(edges, i)
        
        if count2 > best_count:
            best_count = count2
            best_edges = edges2[:]
            
        # Optional: Try swap-based local search if time permits?
        # For now, restarts are cheap.

    return [list(e) for e in best_edges]
