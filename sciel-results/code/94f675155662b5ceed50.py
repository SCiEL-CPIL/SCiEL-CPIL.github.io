import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 145
    n = 16
    
    adj = [[False] * n for _ in range(n)]
    
    # Start with K_{3,13}
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
    
    edge_count = 39
    
    # Precompute all 4-sets
    all_4sets = list(itertools.combinations(range(n), 4))
    
    # For each vertex, precompute 4-sets containing it
    sets_with_vertex = [[] for _ in range(n)]
    for s in all_4sets:
        for v in s:
            sets_with_vertex[v].append(s)
    
    def creates_q3(u, v):
        """Check if adding edge (u,v) creates a Q3. Only check (A,B) with u∈A, v∈B or u∈B, v∈A."""
        adj[u][v] = True
        adj[v][u] = True
        
        found = False
        # Case 1: u∈A, v∈B
        for A in sets_with_vertex[u]:
            A_set = set(A)
            if v in A_set:
                continue
            for B in sets_with_vertex[v]:
                B_set = set(B)
                if A_set & B_set:
                    continue
                # Check K_{4,4} minus perfect matching
                ok = True
                for a in A:
                    if sum(1 for b in B if adj[a][b]) != 3:
                        ok = False
                        break
                if not ok:
                    continue
                for b in B:
                    if sum(1 for a in A if adj[a][b]) != 3:
                        ok = False
                        break
                if not ok:
                    continue
                non_edges = {}
                for a in A:
                    for b in B:
                        if not adj[a][b]:
                            non_edges[a] = b
                            break
                if len(non_edges) == 4 and set(non_edges.values()) == B_set:
                    found = True
                    break
            if found:
                break
        
        # Case 2: u∈B, v∈A
        if not found:
            for B in sets_with_vertex[u]:
                B_set = set(B)
                if v in B_set:
                    continue
                for A in sets_with_vertex[v]:
                    A_set = set(A)
                    if A_set & B_set:
                        continue
                    ok = True
                    for a in A:
                        if sum(1 for b in B if adj[a][b]) != 3:
                            ok = False
                            break
                    if not ok:
                        continue
                    for b in B:
                        if sum(1 for a in A if adj[a][b]) != 3:
                            ok = False
                            break
                    if not ok:
                        continue
                    non_edges = {}
                    for a in A:
                        for b in B:
                            if not adj[a][b]:
                                non_edges[a] = b
                                break
                    if len(non_edges) == 4 and set(non_edges.values()) == B_set:
                        found = True
                        break
                if found:
                    break
        
        adj[u][v] = False
        adj[v][u] = False
        return found
    
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    random.shuffle(non_edges)
    
    for u, v in non_edges:
        if time.time() - start_time > DEADLINE:
            break
        if not creates_q3(u, v):
            adj[u][v] = True
            adj[v][u] = True
            edge_count += 1
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
