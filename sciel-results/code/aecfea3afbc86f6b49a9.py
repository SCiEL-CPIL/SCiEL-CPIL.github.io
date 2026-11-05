import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    Q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    Q3_adj = {i: set() for i in range(8)}
    for u, v in Q3_edges:
        Q3_adj[u].add(v)
        Q3_adj[v].add(u)
    
    def creates_q3(adj, u, v):
        """Check if adding edge (u,v) creates a Q3 in the graph."""
        # Temporarily add the edge
        adj[u][v] = True
        adj[v][u] = True
        
        found = False
        # For each edge in Q3, try to map (u,v) to it
        for eu, ev in Q3_edges:
            if found:
                break
            for (a, b) in [(eu, ev), (ev, eu)]:
                if found:
                    break
                # a -> u, b -> v
                # Backtrack to assign remaining 6 Q3 vertices
                remaining = [x for x in range(8) if x != a and x != b]
                # Order remaining by degree of constraints
                # Vertices adjacent to a or b first
                def constraint_count(x):
                    return sum(1 for y in Q3_adj[x] if y == a or y == b)
                remaining.sort(key=constraint_count, reverse=True)
                
                assignment = {a: u, b: v}
                
                def bt(idx):
                    nonlocal found
                    if time.time() > deadline:
                        found = True  # abort
                        return
                    if idx == len(remaining):
                        # Verify all edges
                        for x, y in Q3_edges:
                            if not adj[assignment[x]][assignment[y]]:
                                return
                        found = True
                        return
                    q3v = remaining[idx]
                    # Candidates: graph vertices not yet used, adjacent to assigned neighbors
                    used = set(assignment.values())
                    for gv in range(n):
                        if gv in used:
                            continue
                        # Check adjacency to already-assigned neighbors
                        ok = True
                        for nb in Q3_adj[q3v]:
                            if nb in assignment:
                                if not adj[gv][assignment[nb]]:
                                    ok = False
                                    break
                        if ok:
                            assignment[q3v] = gv
                            bt(idx + 1)
                            if found:
                                return
                            del assignment[q3v]
                
                bt(0)
        
        # Remove the edge
        adj[u][v] = False
        adj[v][u] = False
        return found
    
    # Start with K_{3,13}
    adj = [[False] * n for _ in range(n)]
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
    
    # Get all missing edges
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline:
            break
        if not creates_q3(adj, u, v):
            adj[u][v] = True
            adj[v][u] = True
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
