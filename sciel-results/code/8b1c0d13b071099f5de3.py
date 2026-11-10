"""Local search: start from Grötzsch core, add edges greedily, then local-search for more."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n = 16

    # Helper: check triangle-free given adjacency matrix
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def can_add_edge(adj, u, v):
        if adj[u][v]:
            return False
        # Adding u-v creates triangle if exists w with adj[u][w] and adj[v][w]
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    # 3-colorability check via backtracking
    def is_3_colorable(adj):
        color = [-1] * n
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda x: -sum(adj[x]))
        
        def bt(idx):
            if idx == n:
                return True
            v = order[idx]
            used = set()
            for u in range(n):
                if adj[u][v] and color[u] != -1:
                    used.add(color[u])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if bt(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return bt(0)

    # Build Grötzsch core (vertices 0..10)
    adj = [[0] * n for _ in range(n)]
    
    # Pentagon 0-4
    for i in range(5):
        u, v = i, (i + 1) % 5
        adj[u][v] = adj[v][u] = 1
    
    # Mycielski: vertices 5..9
    for i in range(5):
        u, v = i, (i + 1) % 5
        adj[u][5 + v] = adj[5 + v][u] = 1
        adj[v][5 + u] = adj[5 + u][v] = 1
    
    # Vertex 10 connected to 5..9
    for u in range(5):
        adj[5 + u][10] = adj[10][5 + u] = 1

    def edge_count(adj):
        return sum(sum(row) for row in adj) // 2

    best_edges = edge_count(adj)
    best_adj = [row[:] for row in adj]

    # Phase 1: Greedy edge addition
    # Try adding all possible edges that don't create triangles
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v] and can_add_edge(adj, u, v):
                    # Check if adding this edge keeps graph non-3-colorable
                    adj[u][v] = adj[v][u] = 1
                    if not is_3_colorable(adj):
                        best_edges = edge_count(adj)
                        best_adj = [row[:] for row in adj]
                        changed = True
                    else:
                        adj[u][v] = adj[v][u] = 0
        if time.time() > deadline - 30:
            break

    # Phase 2: Local search - try to swap edges to increase count
    # Actually, we want to ADD edges. Let's try more aggressive search.
    # Try adding edges in different orders, and also try to find
    # configurations where we can add multiple edges at once.
    
    # Better approach: iterate through all non-edges, try adding each,
    # check constraints. If we can add, add it. Repeat until no more can be added.
    # But also try different starting points / perturbations.
    
    # Let's do multiple restarts with randomized edge addition order
    for restart in range(200):
        if time.time() > deadline - 20:
            break
        
        # Start from current best
        adj = [row[:] for row in best_adj]
        
        # Try adding edges in random order
        non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u][v]]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 20:
                break
            if not adj[u][v] and can_add_edge(adj, u, v):
                adj[u][v] = adj[v][u] = 1
                if not is_3_colorable(adj):
                    ec = edge_count(adj)
                    if ec > best_edges:
                        best_edges = ec
                        best_adj = [row[:] for row in adj]
                else:
                    adj[u][v] = adj[v][u] = 0

    # Phase 3: Try to find better configurations by removing some edges and re-adding
    # This allows escaping local optima
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        adj = [row[:] for row in best_adj]
        
        # Remove a random edge
        edges_in_adj = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
        if not edges_in_adj:
            continue
        u, v = random.choice(edges_in_adj)
        adj[u][v] = adj[v][u] = 0
        
        # Now try to re-add edges greedily in random order
        non_edges = [(a, b) for a in range(n) for b in range(a + 1, n) if not adj[a][b]]
        random.shuffle(non_edges)
        
        for a, b in non_edges:
            if time.time() > deadline - 15:
                break
            if not adj[a][b] and can_add_edge(adj, a, b):
                adj[a][b] = adj[b][a] = 1
                if not is_3_colorable(adj):
                    ec = edge_count(adj)
                    if ec > best_edges:
                        best_edges = ec
                        best_adj = [row[:] for row in adj]
                else:
                    adj[a][b] = adj[b][a] = 0

    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                result.append([u, v])
    
    return result
