def construct():
    import random
    import math

    n = 18
    random.seed(123)

    def is_k4_free(adj):
        # Check if graph is K4-free
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True

    def max_degree(adj):
        degs = [0]*n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    degs[i] += 1
        return max(degs)

    def heuristic_score(adj):
        # Lower bound on T(G)
        degs = [0]*n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    degs[i] += 1
        # T(G) >= max degree
        # T(G) >= independence number (approx)
        return max(degs)

    def construct_greedy(target_max_deg):
        adj = [[False]*n for _ in range(n)]
        # Try to add edges to reach a certain density while keeping max degree low
        # We want to maximize triangles, so we want to add edges that form triangles
        # but avoid K4s
        
        # Start with a random permutation of edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates a K4
            # K4 is formed if there is a pair (a,b) such that 
            # u~a, u~b, v~a, v~b, a~b
            # And u~v is the new edge.
            # So we need to check if there is an edge (a,b) in the current graph
            # such that a,b are common neighbors of u and v.
            
            common = []
            for k in range(n):
                if k != u and k != v:
                    if adj[u][k] and adj[v][k]:
                        common.append(k)
            
            # If there is an edge between any two in common, adding (u,v) creates a K4
            creates_k4 = False
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    if adj[common[i]][common[j]]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if not creates_k4:
                # Check degree constraint
                deg_u = sum(1 for x in range(n) if adj[u][x])
                deg_v = sum(1 for x in range(n) if adj[v][x])
                
                # Heuristic: Allow if degrees are below target
                if deg_u < target_max_deg and deg_v < target_max_deg:
                    adj[u][v] = True
                    adj[v][u] = True
                    
        return adj

    best_edges = None
    best_score = 999
    
    # Try different max degree targets
    # We know T(G) >= max_deg.
    # Turan T(18,3) has max_deg 12.
    # We want to go lower.
    for target_deg in [10, 9, 8]:
        for trial in range(50):
            adj = construct_greedy(target_deg)
            if not is_k4_free(adj):
                continue
            
            score = heuristic_score(adj)
            if score < best_score:
                best_score = score
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            best_edges.append([i, j])

    # Fallback to Turan if nothing found (should not happen)
    if best_edges is None:
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        best_edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        best_edges.append([a, b])

    return best_edges
