"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_k4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
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

def _largest_triangle_free(n, adj):
    """Find the largest subset of vertices inducing a triangle-free subgraph."""
    best = 0
    # Use bitmasks for subsets up to size 18
    # For each subset, check if it's triangle-free
    # Optimize: only need to check subsets of size > best
    # Use recursive approach or iterate by size
    for size in range(n, best, -1):
        found = False
        for subset in combinations(range(n), size):
            # Check if induced subgraph on subset is triangle-free
            is_tf = True
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if adj[subset[i]][subset[j]]:
                        for k in range(j+1, len(subset)):
                            if adj[subset[i]][subset[k]] and adj[subset[j]][subset[k]]:
                                is_tf = False
                                break
                    if not is_tf:
                        break
                if not is_tf:
                    break
            if is_tf:
                found = True
                break
        if found:
            best = size
            break
    return best

def _evaluate(n, adj):
    """Evaluate a K4-free graph: return largest triangle-free induced subgraph size."""
    if not _is_k4_free(n, adj):
        return None
    return _largest_triangle_free(n, adj)

def construct():
    n = 18
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with the known construction as baseline
    best_score = 12
    best_adj = [[0]*n for _ in range(n)]
    # Build K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    best_adj[a][b] = 1
                    best_adj[b][a] = 1
    
    rng = random.Random(42)
    
    # Try random graphs with various densities
    for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for trial in range(200):
            if time.time() > deadline:
                break
            # Generate random graph
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Check K4-free
            if not _is_k4_free(n, adj):
                continue
            
            # Evaluate
            score = _largest_triangle_free(n, adj)
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in adj]
                if best_score <= 9:  # good enough, stop early
                    break
        if time.time() > deadline:
            break
        if best_score <= 9:
            break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
