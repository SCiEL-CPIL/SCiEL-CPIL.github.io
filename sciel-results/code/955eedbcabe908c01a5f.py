"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    _t0 = time.time()
    _DEADLINE = 80.0
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    def is_triangle_free(added):
        for u, v in added:
            if (u - v) % 2 == 0:
                return False
        return True
    
    def compute_diameter(added):
        edge_set = set(base_edges)
        for u, v in added:
            edge_set.add((min(u,v), max(u,v)))
        full_adj = [set() for _ in range(n)]
        for u, v in edge_set:
            full_adj[u].add(v)
            full_adj[v].add(u)
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nb in full_adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            for d in dist:
                if d == -1:
                    return float('inf')
                if d > max_d:
                    max_d = d
            if max_d > 4:
                return max_d
        return max_d

    def time_left():
        return (time.time() - _t0) < _DEADLINE

    # Candidate edges: even-odd pairs not in base
    candidates = []
    cand_index = {}
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    candidates.append((u, v))
                    cand_index[(u, v)] = idx
                    idx += 1
    m = len(candidates)
    
    # Build adjacency for BFS including all candidate edges (for ILP constraint generation)
    # For the ILP, we need: for every pair (s,t), dist(s,t) <= 4
    # We use a flow formulation: for each (s,t), we need a path of length <= 4
    
    # Better approach: use the fact that in a bipartite graph with diameter <= 4,
    # every vertex must be within distance 2 of some "hub" structure.
    # 
    # Alternative: brute force with smart pruning since m is manageable
    # m = number of even-odd non-adjacent pairs
    # Even vertices: 0,2,...,22 (12 vertices), Odd vertices: 1,3,...,23 (12 vertices)
    # Total even-odd pairs: 144, minus 23 base edges = 121 candidates
    
    # For ILP with scipy: minimize sum(x_i) subject to diameter constraints
    # Diameter constraint: for all pairs (s,t), there exists a path of length <= 4
    # This is hard to linearize directly. Instead, use a different approach.
    
    # Key insight: In a bipartite graph, diameter <= 4 means:
    # - For any two vertices s, t, dist(s,t) <= 4
    # - In bipartite graphs, dist is always even or odd depending on partition
    # - Same partition: dist is even (0, 2, 4)
    # - Different partition: dist is odd (1, 3)
    
    # So constraints are:
    # 1. All odd-odd pairs: dist <= 4 (even distance, so 0, 2, or 4)
    # 2. All even-even pairs: dist <= 4 (even distance, so 0, 2, or 4)  
    # 3. All even-odd pairs: dist <= 3 (odd distance, so 1 or 3) -- but wait, diameter <= 4 means dist <= 4
    #    Actually for even-odd, dist must be odd, so dist <= 3 (since 5 > 4)
    
    # Actually, let me reconsider. Diameter <= 4 means ALL pairs have dist <= 4.
    # In bipartite: same-side pairs have even dist, different-side have odd dist.
    # Even dist <= 4: dist in {0, 2, 4}
    # Odd dist <= 4: dist in {1, 3}
    
    # Let me try a smarter search: iterate over possible edge counts and use
    # constraint propagation / backtracking.
    
    # Actually, let's try the ILP approach properly.
    # For each pair (s,t), we need dist(s,t) <= 4.
    # We can model this with: for each (s,t), introduce variables for whether
    # there's a path of length exactly 1, 2, 3, or 4.
    
    # This is complex. Let me instead use a smarter greedy + local search approach
    # that's more likely to find optimal solutions.
    
    # Strategy: 
    # 1. Start with a known valid solution (10 edges)
    # 2. Try to remove edges (already done)
    # 3. Try systematic search for smaller solutions
    
    # Known good: hub 12 with 10 edges
    initial_added = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    if compute_diameter(initial_added) <= 4 and is_triangle_free(initial_added):
        best = initial_added[:]
    else:
        best = initial_added[:]  # fallback
    
    # Try to reduce
    improved = True
    while improved and time_left():
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial
                improved = True
                break

    # Try all single-hub solutions
    for hub in range(0, n, 2):
        if not time_left():
            break
        trial = [(hub, v) for v in range(1, n, 2) if v != hub-1 and v != hub+1]
        if len(trial) < len(best) and is_triangle_free(trial) and compute_diameter(trial) <= 4:
            best = trial

    # Try two-hub solutions
    for h1 in range(0, n, 2):
        if not time_left():
            break
        for h2 in range(h1+2, n, 2):
            if not time_left():
                break
            trial = []
            for v in range(1, n, 2):
                d1 = abs(v - h1)
                d2 = abs(v - h2)
                if d1 <= d2:
                    if v not in (h1-1, h1+1):
                        trial.append((h1, v))
                else:
                    if v not in (h2-1, h2+1):
                        trial.append((h2, v))
            seen = set()
            unique_trial = []
            for e in trial:
                if e not in seen:
                    seen.add(e)
                    unique_trial.append(e)
            if len(unique_trial) < len(best) and is_triangle_free(unique_trial) and compute_diameter(unique_trial) <= 4:
                best = unique_trial

    # Try three-hub solutions
    for h1 in range(0, n, 2):
        if not time_left():
            break
        for h2 in range(h1+2, n, 2):
            if not time_left():
                break
            for h3 in range(h2+2, n, 2):
                if not time_left():
                    break
                trial = []
                for v in range(1, n, 2):
                    d1 = abs(v - h1)
                    d2 = abs(v - h2)
                    d3 = abs(v - h3)
                    dmin = min(d1, d2, d3)
                    if dmin == d1:
                        if v not in (h1-1, h1+1):
                            trial.append((h1, v))
                    elif dmin == d2:
                        if v not in (h2-1, h2+1):
                            trial.append((h2, v))
                    else:
                        if v not in (h3-1, h3+1):
                            trial.append((h3, v))
                seen = set()
                unique_trial = []
                for e in trial:
                    if e not in seen:
                        seen.add(e)
                        unique_trial.append(e)
                if len(unique_trial) < len(best) and is_triangle_free(unique_trial) and compute_diameter(unique_trial) <= 4:
                    best = unique_trial

    # Final reduction pass
    improved = True
    while improved and time_left():
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial
                improved = True
                break
    
    # Fallback
    if not best or not (is_triangle_free(best) and compute_diameter(best) <= 4):
        best = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]

    # Final verification
    base_set = set(base_edges)
    seen = set()
    final = []
    for u, v in best:
        a, b = min(u, v), max(u, v)
        if (a, b) in base_set or (a, b) in seen:
            continue
        seen.add((a, b))
        final.append((a, b))
    if not (is_triangle_free(final) and compute_diameter(final) <= 4):
        final = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]

    result = [[int(u), int(v)] for u, v in final]
    return result
# EVOLVE-BLOCK-END
