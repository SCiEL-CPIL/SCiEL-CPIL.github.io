"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

# Precompute edge position lookup
_edge_pos = {}
for i, e in enumerate(P['edge_order']):
    _edge_pos[(min(e[0], e[1]), max(e[0], e[1]))] = i + 1  # 1-based

def _cycle_score(cycle):
    """Compute the score of a Hamilton cycle: max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        key = (min(u, v), max(u, v))
        pos = _edge_pos[key]
        if pos > max_pos:
            max_pos = pos
    return max_pos

def _find_hamilton_cycle_in_graph(adj, n, time_limit=5.0):
    """Find a Hamilton cycle in a graph given by adjacency list using backtracking."""
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    deadline = time.time() + time_limit
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[path[-1]]:
                return path[:]
            return None
        
        for v in adj[path[-1]]:
            if not visited[v]:
                # Pruning: if v is not start and v is the last unvisited vertex,
                # check if it can connect back to start
                if len(path) == n - 1 and start not in adj[v]:
                    continue
                visited[v] = True
                path.append(v)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited[v] = False
        return None
    
    return backtrack()

def _two_opt_improve(cycle, adj_threshold, max_iter=10000):
    """Improve a cycle using 2-opt moves, only using edges with position <= threshold."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment i+1..j
                u = cycle[i]
                v = cycle[i + 1]
                w = cycle[j]
                x = cycle[(j + 1) % n]
                
                # Check if new edges (u,w) and (v,x) exist in threshold graph
                key1 = (min(u, w), max(u, w))
                key2 = (min(v, x), max(v, x))
                if key1 not in adj_threshold or key2 not in adj_threshold:
                    continue
                
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
    
    return best, best_score

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80.0  # Leave margin before 90s kill
    
    # Start with identity permutation
    best_cycle = list(range(n))
    best_score = _cycle_score(best_cycle)
    
    # Binary search for minimum threshold K such that G_K contains a Hamilton cycle
    # G_K has edges with position <= K
    lo, hi = n, len(P['edge_order'])  # Minimum possible is n (a cycle has n edges)
    
    # First, find an upper bound by trying increasing thresholds
    # Use a greedy approach to quickly find a feasible cycle
    
    # Strategy: For each candidate threshold K, check if G_K has a Hamilton cycle
    # Use backtracking with pruning
    
    # Start with a reasonable initial guess
    # The identity cycle has some score; let's use that as upper bound
    current_hi = best_score
    
    # Try to find lower thresholds
    for K in range(n, current_hi + 1):
        if time.time() > deadline - 10.0:
            break
        
        # Build adjacency list for edges with position <= K
        adj = [set() for _ in range(n)]
        for i, e in enumerate(P['edge_order']):
            if i + 1 > K:
                break
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if every vertex has degree >= 2
        if any(len(adj[v]) < 2 for v in range(n)):
            continue
        
        # Try to find Hamilton cycle using backtracking
        cycle = _find_hamilton_cycle_in_graph(adj, n, time_limit=3.0)
        if cycle is not None:
            score = _cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
                current_hi = K  # Update upper bound
                # Don't break; keep searching for even lower K
    
    # Now do local improvement on the best cycle found
    if time.time() < deadline - 5.0:
        # Use 2-opt with the threshold graph
        K = best_score
        adj_threshold = set()
        for i, e in enumerate(P['edge_order']):
            if i + 1 <= K:
                adj_threshold.add((min(e[0], e[1]), max(e[0], e[1])))
        
        improved_cycle, improved_score = _two_opt_improve(best_cycle, adj_threshold, max_iter=5000)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = improved_cycle
    
    # Final 2-opt polish without threshold restriction
    if time.time() < deadline - 2.0:
        # Unrestricted 2-opt
        n = len(best_cycle)
        for _ in range(2000):
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    u = best_cycle[i]
                    v = best_cycle[i + 1]
                    w = best_cycle[j]
                    x = best_cycle[(j + 1) % n]
                    
                    new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                    new_score = _cycle_score(new_cycle)
                    if new_score < best_score:
                        best_score = new_score
                        best_cycle = new_cycle
                        improved = True
            if not improved:
                break
    
    return best_cycle
# EVOLVE-BLOCK-END
