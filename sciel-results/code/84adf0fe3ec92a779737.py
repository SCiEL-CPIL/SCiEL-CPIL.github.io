import random
import time

def has_k4(edges, n):
    # Build adjacency set for O(1) lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check all combinations of 4 vertices
    # Optimization: Only check if potential
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]: continue
            for w in range(v+1, n):
                if w not in adj[u] or w not in adj[v]: continue
                for x in range(w+1, n):
                    if x in adj[u] and x in adj[v] and x in adj[w]:
                        return True
    return False

def has_independent_set_6(edges, n):
    # Build non-adjacency (complement) adjacency
    # Independent set in G is clique in G_comp
    # G_comp edge (u,v) if NOT in G
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                comp_adj[u].add(v)
                comp_adj[v].add(u)
                
    # Find clique of size 6 in comp_adj
    # Backtracking
    def find_clique(vertices, size, current):
        if size == 0:
            return True
        if len(vertices) < size:
            return False
        if size == 1:
            return True
            
        v = vertices[0]
        rest = vertices[1:]
        
        # Include v
        neighbors = [u for u in rest if u in comp_adj[v]]
        if find_clique(neighbors, size-1, current + [v]):
            return True
            
        # Exclude v
        if find_clique(rest, size, current):
            return True
            
        return False

    return find_clique(list(range(n)), 6, [])

def construct():
    n = 35
    # Initialize with a random graph
    # We want to find H (complement of G) with alpha(H) <= 3 and omega(H) <= 5
    # So G has no K4 (omega(G)<=3) and no alpha 6 (alpha(G)<=5)
    
    # Let's work directly on G
    # G must have no K4 and no independent set of 6.
    
    # Strategy: Local search on G
    # Start with a random graph
    random.seed(42)
    adj = [[False]*n for _ in range(n)]
    edges = []
    
    # Initialize with a random regular-ish graph
    # Degree ~ 8-10
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.25: # ~8.5 degree
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
                
    def get_edges_from_adj():
        e = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    e.append([i, j])
        return e
        
    def find_violation():
        # Returns (type, vertices)
        # Type 0: K4, Type 1: Independent Set 6
        # Check K4
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if not adj[u][w] or not adj[v][w]: continue
                    for x in range(w+1, n):
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            return (0, [u, v, w, x])
        # Check Independent Set 6 (Clique in complement)
        # Complement adjacency: not adj[u][v]
        def find_clique_in_comp(vertices, size, current):
            if size == 0:
                return current
            if len(vertices) < size:
                return None
            if size == 1:
                return current + [vertices[0]]
                
            v = vertices[0]
            rest = vertices[1:]
            
            # Include v
            # Neighbors in complement: u in rest such that not adj[v][u]
            neighbors = [u for u in rest if not adj[v][u]]
            res = find_clique_in_comp(neighbors, size-1, current + [v])
            if res:
                return res
            
            # Exclude v
            return find_clique_in_comp(rest, size, current)
            
        clique = find_clique_in_comp(list(range(n)), 6, [])
        if clique:
            return (1, clique)
        return None

    start_time = time.time()
    
    while time.time() - start_time < 110:
        viol = find_violation()
        if not viol:
            break
        v_type, verts = viol
        if v_type == 0:
            # Found K4, remove an edge
            # Pick a random edge from the K4
            u, v, w, x = verts
            candidates = [[u,v], [u,w], [u,x], [v,w], [v,x], [w,x]]
            chosen = random.choice(candidates)
            a, b = chosen
            adj[a][b] = False
            adj[b][a] = False
        else:
            # Found Independent Set 6, add an edge
            # Pick a random non-edge from the set
            verts_set = set(verts)
            # All pairs are non-edges in G
            pairs = [(i, j) for i in range(len(verts)) for j in range(i+1, len(verts))]
            i, j = random.choice(pairs)
            a, b = verts[i], verts[j]
            adj[a][b] = True
            adj[b][a] = True
            
    # Final check
    edges = get_edges_from_adj()
    if not has_k4(edges, n) and not has_independent_set_6(edges, n):
        return {'n': n, 'edges': edges}
        
    # Fallback: Try a few more random restarts if failed
    for _ in range(5):
        random.seed(random.randint(0, 10000))
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.25:
                    adj[i][j] = True
                    adj[j][i] = True
                    
        start_time = time.time()
        while time.time() - start_time < 10:
            viol = find_violation()
            if not viol:
                break
            v_type, verts = viol
            if v_type == 0:
                u, v, w, x = verts
                candidates = [[u,v], [u,w], [u,x], [v,w], [v,x], [w,x]]
                chosen = random.choice(candidates)
                a, b = chosen
                adj[a][b] = False
                adj[b][a] = False
            else:
                verts_set = set(verts)
                pairs = [(i, j) for i in range(len(verts)) for j in range(i+1, len(verts))]
                i, j = random.choice(pairs)
                a, b = verts[i], verts[j]
                adj[a][b] = True
                adj[b][a] = True
                
        edges = get_edges_from_adj()
        if not has_k4(edges, n) and not has_independent_set_6(edges, n):
            return {'n': n, 'edges': edges}
            
    # If all fails, return a smaller valid graph (e.g. Paley 17)
    # But we expect 35 to work.
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}
