"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _build_adj(n, edges):
    """Build adjacency matrix from edge list."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _count_k4(adj, n):
    """Count number of K4 subgraphs."""
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        count += 1
    return count

def _count_ind6(adj, n):
    """Count number of independent 6-sets."""
    count = 0
    non_adj = [[not adj[i][j] for j in range(n)] for i in range(n)]
    for combo in itertools.combinations(range(n), 6):
        valid = True
        for i in range(6):
            for j in range(i + 1, 6):
                if not non_adj[combo[i]][combo[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            count += 1
    return count

def _is_valid(adj, n):
    """Check if graph is K4-free and has no independent 6-set."""
    # Check K4
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    # Check independent 6-set
    non_adj = [[not adj[i][j] for j in range(n)] for i in range(n)]
    for combo in itertools.combinations(range(n), 6):
        valid = True
        for i in range(6):
            for j in range(i + 1, 6):
                if not non_adj[combo[i]][combo[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return False
    return True

def _find_k4_violation(adj, n):
    """Find a K4 subgraph, return tuple of 4 vertices or None."""
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return (a, b, c, d)
    return None

def _find_ind6_violation(adj, n):
    """Find an independent 6-set, return tuple of 6 vertices or None."""
    non_adj = [[not adj[i][j] for j in range(n)] for i in range(n)]
    for combo in itertools.combinations(range(n), 6):
        valid = True
        for i in range(6):
            for j in range(i + 1, 6):
                if not non_adj[combo[i]][combo[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return combo
    return None

def _random_graph(n, p):
    """Generate a random graph with edge probability p."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges.append([i, j])
    return edges

def _flip_edge(adj, n, i, j):
    """Flip edge (i,j) in adjacency matrix."""
    adj[i][j] = not adj[i][j]
    adj[j][i] = not adj[j][i]

def _edges_from_adj(adj, n):
    """Extract edge list from adjacency matrix."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_n = 17
    best_edges = None
    
    # Start with Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    
    # Try increasing n from 18 upward
    for n in range(18, 36):
        if time.time() > deadline - 30.0:
            break
        
        # Multiple random restarts
        found = False
        for attempt in range(50):
            if time.time() > deadline - 25.0:
                break
            
            # Start with random graph with appropriate density
            # For K4-free, we want relatively sparse graphs
            # For no independent 6-set, we want relatively dense graphs
            p = 0.5
            edges = _random_graph(n, p)
            adj = _build_adj(n, edges)
            
            # Local search: fix violations by flipping edges
            for step in range(2000):
                if time.time() > deadline - 20.0:
                    break
                
                # Check for K4 violation
                k4 = _find_k4_violation(adj, n)
                if k4 is not None:
                    # Flip a random edge within the K4 to break it
                    verts = list(k4)
                    random.shuffle(verts)
                    i, j = verts[0], verts[1]
                    _flip_edge(adj, n, i, j)
                    continue
                
                # Check for independent 6-set violation
                ind6 = _find_ind6_violation(adj, n)
                if ind6 is not None:
                    # Flip a random edge within the independent set to break it
                    verts = list(ind6)
                    random.shuffle(verts)
                    i, j = verts[0], verts[1]
                    _flip_edge(adj, n, i, j)
                    continue
                
                # No violations found - graph is valid!
                edges = _edges_from_adj(adj, n)
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    found = True
                break
            
            if found:
                break
        
        if not found:
            # If we couldn't find a valid graph for this n, try next
            continue
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
