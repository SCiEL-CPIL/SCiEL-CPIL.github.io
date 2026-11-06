import time
import random
from itertools import combinations

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is triangle-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using backtracking."""
    color = [-1] * n
    deg = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        for c in range(3):
            ok = True
            for u in range(n):
                if adj[v][u] and color[u] == c:
                    ok = False
                    break
            if ok:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return not backtrack(0)

def _count_edges(adj, n):
    return sum(sum(adj[i]) for i in range(n)) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """Create adjacency matrix for Grötzsch graph on vertices 0-10."""
    adj = [[0]*n for _ in range(n)]
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    # Vertex 10
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_max_independent_sets(adj, n, count, size):
    """Find 'count' independent sets of size 'size' in the graph."""
    sets = []
    vertices = list(range(n))
    
    def is_independent(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True

    # Simple search: generate combinations
    # For n=11, size=5, C(11,5) = 462. Very fast.
    for comb in combinations(vertices, size):
        if is_independent(comb):
            sets.append(list(comb))
            if len(sets) >= count:
                break
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    adj = _groetzsch_adj(n)
    
    # Find 5 independent sets of size 5 in the Grötzsch part (0-10)
    iso_sets = _find_max_independent_sets(adj, 11, 5, 5)
    
    # If we don't have 5, we might have fewer. 
    # But Grötzsch has 5 max independent sets? 
    # Actually, it has more. 
    # If we have fewer than 5, we can use smaller sets or repeat?
    # Repeating is fine for triangle-freeness.
    
    if len(iso_sets) < 5:
        # Fallback: use the same set or smaller ones
        # For now, assume we found 5. If not, pad with empty or smaller.
        while len(iso_sets) < 5:
            iso_sets.append(iso_sets[0] if iso_sets else [])

    # Connect new vertices 11-15 to the independent sets
    for i in range(5):
        v_new = 11 + i
        for u in iso_sets[i]:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Local Search
    # Try to add edges
    for iter in range(1000):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle non-edges to avoid bias
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            # Check triangle
            creates_triangle = False
            for k in range(n):
                if k != i and k != j and adj[i][k] and adj[j][k]:
                    creates_triangle = True
                    break
            if creates_triangle:
                continue
            
            # Tentatively add
            adj[i][j] = 1
            adj[j][i] = 1
            
            if _is_not_3_colorable(adj, n):
                # Keep
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
            else:
                # Remove
                adj[i][j] = 0
                adj[j][i] = 0
        
        if not improved:
            break
            
    return _get_edges(best_adj, n)
