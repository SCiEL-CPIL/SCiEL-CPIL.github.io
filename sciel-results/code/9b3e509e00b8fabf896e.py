"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import defaultdict

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _build_edge_index():
    n = P['n']
    edge_idx = {}
    for i, (u, v) in enumerate(P['edge_order']):
        edge_idx[(min(u,v), max(u,v))] = i + 1  # 1-based
    return edge_idx

def _cycle_score(cycle, edge_idx):
    """Compute the max edge arrival index for a Hamilton cycle given as a list of vertices."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i+1) % n]
        key = (min(u,v), max(u,v))
        idx = edge_idx[key]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _greedy_cycle(edge_idx, start=0):
    """Build a greedy Hamilton cycle starting from a given vertex."""
    n = P['n']
    visited = [False] * n
    visited[start] = True
    cycle = [start]
    current = start
    
    for _ in range(n - 1):
        best_next = -1
        best_idx = float('inf')
        for v in range(n):
            if not visited[v]:
                key = (min(current, v), max(current, v))
                idx = edge_idx[key]
                if idx < best_idx:
                    best_idx = idx
                    best_next = v
        if best_next == -1:
            # Should not happen in complete graph
            for v in range(n):
                if not visited[v]:
                    best_next = v
                    break
        visited[best_next] = True
        cycle.append(best_next)
        current = best_next
    
    return cycle

def _two_opt(cycle, edge_idx, max_iters=10000):
    """Improve a Hamilton cycle using 2-opt style moves minimizing max edge index."""
    n = len(cycle)
    current_score = _cycle_score(cycle, edge_idx)
    best_cycle = list(cycle)
    best_score = current_score
    
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if j == i + 1 or (i == 0 and j == n-1):
                    continue
                # Try reversing segment i..j
                new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                new_score = _cycle_score(new_cycle, edge_idx)
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    improved = True
        if not improved:
            break
        cycle = best_cycle
    
    return best_cycle, best_score

def _random_local_search(edge_idx, n_restarts=50, time_budget=50):
    """Run multiple random restarts with local search."""
    import time
    start_time = time.time()
    n = P['n']
    best_cycle = None
    best_score = float('inf')
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initial cycle
        cycle = list(range(n))
        rng.shuffle(cycle)
        
        # Greedy improvement from this start
        # First do a greedy pass to get a decent starting point
        # Then local search
        cycle, score = _two_opt(cycle, edge_idx, max_iters=5000)
        
        if score < best_score:
            best_score = score
            best_cycle = list(cycle)
    
    return best_cycle, best_score

def _greedy_with_lookahead(edge_idx):
    """Greedy construction with lookahead to avoid trapping."""
    n = P['n']
    visited = [False] * n
    visited[0] = True
    cycle = [0]
    current = 0
    
    for step in range(n - 1):
        candidates = []
        for v in range(n):
            if not visited[v]:
                key = (min(current, v), max(current, v))
                idx = edge_idx[key]
                candidates.append((idx, v))
        
        candidates.sort()
        
        # Try each candidate, prefer the one that leads to best completion
        best_next = candidates[0][1]
        for idx, v in candidates[:3]:  # Look at top 3
            # Quick check: can we still complete?
            # In complete graph, always yes, but prefer edges that are early
            # Just pick the earliest edge
            best_next = v
            break
        
        visited[best_next] = True
        cycle.append(best_next)
        current = best_next
    
    return cycle

# Precompute edge index
_edge_idx = _build_edge_index()

def construct():
    n = P['n']
    
    # Try greedy from vertex 0
    greedy_cycle = _greedy_cycle(_edge_idx, start=0)
    greedy_score = _cycle_score(greedy_cycle, _edge_idx)
    
    # Try local search from greedy
    ls_cycle, ls_score = _two_opt(greedy_cycle, _edge_idx, max_iters=20000)
    
    # Try random local search
    rs_cycle, rs_score = _random_local_search(_edge_idx, n_restarts=30, time_budget=45)
    
    # Pick the best
    best_cycle = ls_cycle
    best_score = ls_score
    if rs_score < best_score:
        best_cycle = rs_cycle
        best_score = rs_score
    if greedy_score < best_score:
        best_cycle = greedy_cycle
        best_score = greedy_score
    
    return best_cycle
# EVOLVE-BLOCK-END
