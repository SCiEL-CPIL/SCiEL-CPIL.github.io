"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _has_k4(adj, n):
    """Check if graph has K4 using adjacency matrix."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return True
    return False

def _find_k4(adj, n):
    """Find a K4 in the graph, return (a,b,c,d) or None."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return (a, b, c, d)
    return None

def _has_ind6(adj, n):
    """Check if graph has independent set of size 6."""
    for quad in combinations(range(n), 6):
        valid = True
        for i in range(6):
            for j in range(i+1, 6):
                if adj[quad[i]][quad[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return True
    return False

def _find_ind6(adj, n):
    """Find an independent set of size 6, return tuple or None."""
    for quad in combinations(range(n), 6):
        valid = True
        for i in range(6):
            for j in range(i+1, 6):
                if adj[quad[i]][quad[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return quad
    return None

def _is_valid(adj, n):
    """Check if graph is valid (no K4, no independent 6-set)."""
    return not _has_k4(adj, n) and not _has_ind6(adj, n)

def _greedy_build(n, seed=42):
    """Build a graph greedily, trying to avoid K4 and ind6."""
    random.seed(seed)
    adj = [[False]*n for _ in range(n)]
    
    # Start with random sparse graph
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.3:
                adj[i][j] = adj[j][i] = True
    
    # Local search: fix violations
    for iteration in range(500):
        k4 = _find_k4(adj, n)
        if k4:
            # Remove one edge from the K4
            a, b, c, d = k4
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            random.shuffle(edges)
            u, v = edges[0]
            adj[u][v] = adj[v][u] = False
            continue
        
        ind6 = _find_ind6(adj, n)
        if ind6:
            # Add an edge to break the independent set
            quad = ind6
            pairs = list(combinations(quad, 2))
            random.shuffle(pairs)
            u, v = pairs[0]
            # Only add if it doesn't create K4
            adj[u][v] = adj[v][u] = True
            if _has_k4(adj, n):
                adj[u][v] = adj[v][u] = False
            continue
        
        # No violations, try to improve by adding edges
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    adj[i][j] = adj[j][i] = True
                    if _has_k4(adj, n) or _has_ind6(adj, n):
                        adj[i][j] = adj[j][i] = False
                    else:
                        added = True
                        break
            if added:
                break
        
        if not added:
            break
    
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_adj = None
    best_n = 1
    
    # Try increasing n from small to large
    for n in range(17, 36):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            adj = _greedy_build(n, seed)
            
            if _is_valid(adj, n):
                best_adj = adj
                best_n = n
                break
        else:
            # If we couldn't find valid graph for this n, try next
            continue
        
        # If we found valid graph for this n, try to go higher
        if best_n == n:
            continue
        else:
            break
    
    # If no good graph found, fall back to Paley
    if best_adj is None:
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % 17 in squares:
                    adj[i][j] = adj[j][i] = True
        best_adj = adj
        best_n = n
    
    edges = [[i, j] for i in range(best_n) for j in range(i+1, best_n) if best_adj[i][j]]
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
