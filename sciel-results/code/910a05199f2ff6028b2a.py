import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin from 120s
    
    # Helper to check if adding edge (u, v) creates a C4
    def can_add_edge(adj, u, v):
        if u == v:
            return False
        if adj[u][v]:
            return True # Already exists, no change
        
        # Check if there is a path u-a-b-v
        # This is equivalent to checking if there is an edge between N(u) and N(v)
        # Note: if a is in N(u) and N(v), it's a common neighbor.
        # If (a, b) is an edge with a in N(u), b in N(v), then u-a-b-v-u is a C4.
        
        # Optimization: Iterate over the smaller neighborhood
        nu = [x for x in range(len(adj)) if adj[u][x]]
        nv = [x for x in range(len(adj)) if adj[v][x]]
        
        if len(nu) > len(nv):
            nu, nv = nv, nu
            
        for a in nu:
            # If a is also in nv, it's a common neighbor.
            # This doesn't immediately create a C4 by itself, 
            # but if there is ANOTHER path... 
            # Actually, the condition "no two vertices have two common neighbors" 
            # is the standard definition for C4-free.
            # Let's re-verify.
            # If u and v have two common neighbors w1, w2, then u-w1-v-w2-u is a C4.
            # So, we must ensure |N(u) intersect N(v)| <= 1.
            # BUT, this is for the CURRENT graph.
            # When we add (u,v), N(u) and N(v) change.
            # The new common neighbors of u and v are the OLD common neighbors.
            # So if |N(u) intersect N(v)| >= 1, adding (u,v) creates a triangle, which is fine.
            # But does it create a C4?
            # A C4 involves 4 distinct vertices.
            # If we add (u,v), a C4 is formed if there is a path of length 3 between u and v.
            # Path u-a-b-v.
            # This requires a in N(u), b in N(v), and (a,b) in E.
            # So we check: Is there an edge between N(u) and N(v)?
            
            for b in nv:
                if adj[a][b]:
                    return False
        return True

    def find_independent_set(adj, size):
        # Greedy with randomization to find an independent set of given size
        n = len(adj)
        candidates = list(range(n))
        random.shuffle(candidates)
        independent_set = []
        
        for v in candidates:
            if v not in independent_set:
                # Check if v is adjacent to any in independent_set
                is_independent = True
                for u in independent_set:
                    if adj[v][u]:
                        is_independent = False
                        break
                if is_independent:
                    independent_set.append(v)
                    if len(independent_set) >= size:
                        return independent_set
        return independent_set

    def has_independent_set_of_size(adj, size):
        # Try a few random restarts to find an independent set
        for _ in range(50):
            iset = find_independent_set(adj, size)
            if len(iset) >= size:
                return iset
        return None

    def local_search(n, max_iters=2000):
        # Initialize random C4-free graph
        adj = [[False] * n for _ in range(n)]
        
        # Greedy initialization: add random edges if they don't create C4
        vertices = list(range(n))
        random.shuffle(vertices)
        # To make it denser, we can try all pairs in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add_edge(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
        
        # Local Search
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Check if valid
            iset = has_independent_set_of_size(adj, 11)
            if not iset:
                return adj
            
            # Try to add an edge within the independent set
            added = False
            # Randomize order of pairs in iset
            pairs_in_set = [(u, v) for i, u in enumerate(iset) for v in iset[i+1:]]
            random.shuffle(pairs_in_set)
            
            for u, v in pairs_in_set:
                if can_add_edge(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    added = True
                    break
                    
            if not added:
                # Stuck. Perturb: remove a random edge? Or restart?
                # Let's try removing a random edge to allow new configurations
                # Or just break and restart the whole process
                break
                
        return None

    best_result = None
    
    # Try n from 39 down to 30
    for n in range(39, 29, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        found = False
        # Multiple restarts for each n
        for restart in range(20):
            if time.time() - start_time > time_limit - 5:
                break
                
            adj = local_search(n, max_iters=500)
            if adj is not None:
                # Verify
                iset = has_independent_set_of_size(adj, 11)
                if not iset:
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if adj[u][v]:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                else:
                    # It claimed to be valid but has an independent set?
                    # This can happen if has_independent_set_of_size is incomplete (false negative)
                    # But we used it to STOP local search. 
                    # If local search stops, it means it couldn't reduce alpha.
                    # So adj is NOT valid.
                    pass

    # Fallback: The seed solution
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

