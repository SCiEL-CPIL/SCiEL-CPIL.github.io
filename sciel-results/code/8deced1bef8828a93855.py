import random
import time

def construct():
    n = 20
    # Initialize with complete graph
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = 1
            adj[j][i] = 1
            
    def find_k33():
        # Returns a list of edges in a K3,3 if found, else None
        # Iterate over all pairs of disjoint triples
        # Optimization: precompute triples?
        # For n=20, 1140 triples. Pairs of disjoint triples ~ 387k.
        # This is fast enough to run in a loop.
        
        # To speed up, we can check specific structures.
        # But for n=20, brute force is fine.
        
        # We need to return one K3,3.
        # We can randomize the order of checking to avoid bias.
        
        # Generate all triples
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    triples.append((i, j, k))
                    
        # Shuffle to randomize
        random.shuffle(triples)
        
        for t1 in triples:
            # Check if t1 is fully connected? No, K3,3 is bipartite.
            # The edges of K3,3 are between t1 and t2.
            # We need t1 and t2 to be disjoint.
            # And all edges between t1 and t2 must exist.
            
            # Optimization: If t1 does not have enough potential neighbors, skip?
            # For each v in t1, count neighbors in the rest of the graph.
            # This is getting complicated. Let's just check.
            
            for t2 in triples:
                # Check disjoint
                if len(set(t1) & set(t2)) > 0:
                    continue
                    
                # Check if all 9 edges exist
                valid = True
                for u in t1:
                    for v in t2:
                        if adj[u][v] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Found K3,3
                    # Return edges
                    edges = []
                    for u in t1:
                        for v in t2:
                            edges.append((u, v))
                    return edges
                    
        return None

    # Repair loop
    while True:
        k33_edges = find_k33()
        if k33_edges is None:
            break
        # Remove a random edge from k33_edges
        # Or the one that appears in most K3,3s?
        # Let's just pick random for now to keep it simple and avoid expensive counting
        e = random.choice(k33_edges)
        u, v = e
        adj[u][v] = 0
        adj[v][u] = 0

    # Try to add edges
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 0:
                    # Check if adding (i,j) creates K3,3
                    # A K3,3 containing (i,j) must have i and j in different parts?
                    # No, i and j are connected. In K3,3, edges are between parts.
                    # So if we add (i,j), i and j must be in different parts of the K3,3.
                    # Let A be the part containing i, B containing j.
                    # |A|=3, |B|=3.
                    # i in A, j in B.
                    # A \ {i} must be 2 vertices connected to all 3 in B.
                    # B \ {j} must be 2 vertices connected to all 3 in A.
                    # And i connected to all B, j connected to all A.
                    
                    # Candidates for A: i + 2 neighbors of i.
                    # Candidates for B: j + 2 neighbors of j.
                    
                    # This check is specific.
                    # Let's use a general check:
                    # Temporarily add edge, check for K3,3 involving this edge.
                    # If found, remove.
                    
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
                    # Check if this specific edge is in a K3,3
                    # We can just run find_k33 and see if it returns an edge set containing (i,j)?
                    # That's slow.
                    
                    # Better:
                    # For each pair of vertices x, y such that x != i, x != j, y != i, y != j
                    # and x, y not i, j?
                    
                    # Let's stick to the simple repair:
                    # If adding creates K3,3, remove it.
                    # We can check:
                    # Does there exist a set A of size 3 containing i, and B of size 3 containing j,
                    # disjoint, such that all edges between A and B exist?
                    
                    # Neighbors of i: N(i)
                    # Neighbors of j: N(j)
                    
                    # A must be subset of {i} U N(i) ? No, A is just a set of vertices.
                    # But all edges between A and B must exist.
                    # So for u in A, v in B, adj[u][v] must be 1.
                    
                    # Since i is in A and j is in B, adj[i][j] is 1 (we just added it).
                    # For other u in A \ {i}, adj[u][j] must be 1. So u in N(j).
                    # For other v in B \ {j}, adj[i][v] must be 1. So v in N(i).
                    
                    # So A \ {i} must be a subset of N(j) \ {i}.
                    # And B \ {j} must be a subset of N(i) \ {j}.
                    
                    # Let C = N(j) \ {i, j}
                    # Let D = N(i) \ {i, j}
                    
                    # We need to pick 2 vertices from C (call them x, y) to form A = {i, x, y}.
                    # And 2 vertices from D (call them u, v) to form B = {j, u, v}.
                    # Conditions:
                    # 1. A and B are disjoint.
                    # 2. Edges between A and B exist.
                    #    - i-u, i-v, i-j (ok), x-u, x-v, x-j (ok since x in N(j)), y-u, y-v, y-j (ok).
                    #    - So we need x-u, x-v, y-u, y-v to exist.
                    
                    # So we need to find x, y in C and u, v in D such that:
                    # {i, x, y} intersect {j, u, v} is empty.
                    # And x, y connected to u, v.
                    
                    # This is a small subgraph check.
                    
                    found_k33 = False
                    # Iterate over pairs in C
                    C = [x for x in range(n) if adj[j][x] == 1 and x != i and x != j]
                    D = [v for v in range(n) if adj[i][v] == 1 and v != i and v != j]
                    
                    # We need x, y in C and u, v in D.
                    # To avoid duplicates, we can iterate combinations.
                    # But C and D are small (max 18).
                    # Pairs in C: ~ 150. Pairs in D: ~ 150. Total 22500 checks.
                    # This is acceptable.
                    
                    import itertools
                    for x, y in itertools.combinations(C, 2):
                        A = {i, x, y}
                        for u, v in itertools.combinations(D, 2):
                            B = {j, u, v}
                            if A & B:
                                continue
                            # Check edges between A and B
                            # i-u, i-v
                            if adj[i][u] == 0 or adj[i][v] == 0: continue
                            # x-j, y-j (guaranteed by C)
                            # x-u, x-v, y-u, y-v
                            if adj[x][u] == 1 and adj[x][v] == 1 and adj[y][u] == 1 and adj[y][v] == 1:
                                found_k33 = True
                                break
                        if found_k33:
                            break
                            
                    if found_k33:
                        # Revert
                        adj[i][j] = 0
                        adj[j][i] = 0
                    else:
                        improved = True

    # Return edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 1:
                edges.append([i, j])
    return edges
