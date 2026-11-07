"""Rewritten program: Greedy Set Cover with Distance Matrix for triangle-free diameter-4 augmentation."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# Precompute bipartition
_EVEN = [i for i in range(_N) if i % 2 == 0]
_ODD = [i for i in range(_N) if i % 2 == 1]

# All valid candidate edges (bipartite, non-base)
_CANDIDATES = []
for u in _EVEN:
    for v in _ODD:
        if (u, v) not in _BASE_SET and (v, u) not in _BASE_SET:
            _CANDIDATES.append((u, v))

def _init_dist_matrix():
    """Initialize distance matrix with path distances."""
    dist = [[0] * _N for _ in range(_N)]
    for i in range(_N):
        for j in range(_N):
            dist[i][j] = abs(i - j)
    return dist

def _update_dist_matrix(dist, u, v):
    """Update distance matrix after adding edge (u, v).
    Since the graph is unweighted and we are adding edges, 
    we can use a simple relaxation step.
    For all i, j: dist[i][j] = min(dist[i][j], dist[i][u] + 1 + dist[v][j], dist[i][v] + 1 + dist[u][j])
    However, this single pass might not propagate fully if multiple new shortcuts interact.
    But since we are adding one edge at a time, and the graph is small, 
    a few iterations of relaxation or a full Floyd-Warshall is safe.
    Given N=24, Floyd-Warshall is O(N^3) = 13k ops. Very fast.
    """
    # Run Floyd-Warshall from scratch for simplicity and correctness, 
    # or just update using the new edge as a potential intermediate.
    # Actually, just running Bellman-Ford or Floyd on the updated graph is safest.
    # Let's just recompute from the adjacency representation implicitly?
    # No, we have the matrix. Let's do a full Floyd-Warshall pass.
    # It's O(N^3). 24^3 is tiny.
    for k in range(_N):
        for i in range(_N):
            dik = dist[i][k]
            for j in range(_N):
                nd = dik + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd

def _get_max_dist(dist):
    max_d = 0
    for i in range(_N):
        for j in range(_N):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _count_benefits(dist, u, v, target_diam):
    """Count how many pairs (i, j) with dist[i][j] > target_diam 
    would have dist <= target_diam if edge (u, v) were added."""
    benefits = 0
    for i in range(_N):
        for j in range(_N):
            if i == j: continue
            if dist[i][j] > target_diam:
                # Check if new path via u-v is shorter
                # Path 1: i -> u -> v -> j
                # Path 2: i -> v -> u -> j
                new_d1 = dist[i][u] + 1 + dist[v][j]
                new_d2 = dist[i][v] + 1 + dist[u][j]
                if min(new_d1, new_d2) <= target_diam:
                    benefits += 1
    return benefits

def _construct_greedy_set_cover(seed):
    """Greedy Set Cover approach."""
    rng = random.Random(seed)
    dist = _init_dist_matrix()
    added = []
    added_set = set()
    target = P['diameter']
    
    while _get_max_dist(dist) > target:
        best_edge = None
        max_benefit = -1
        
        # Shuffle candidates for tie-breaking diversity
        cands = list(_CANDIDATES)
        rng.shuffle(cands)
        
        for (u, v) in cands:
            if (u, v) in added_set:
                continue
            benefit = _count_benefits(dist, u, v, target)
            if benefit > max_benefit:
                max_benefit = benefit
                best_edge = (u, v)
        
        if best_edge is None or max_benefit == 0:
            # No single edge helps? This shouldn't happen if graph is connected and target is reachable
            # But if we are stuck, break
            break
            
        u, v = best_edge
        added.append((u, v))
        added_set.add((u, v))
        
        # Update distance matrix
        _update_dist_matrix(dist, u, v)
        
    return added

def _try_remove_edges(added, target_diam):
    """Try to remove redundant edges."""
    cur = list(added)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            # Try removing cur[i]
            trial = cur[:i] + cur[i+1:]
            # Check validity
            # Rebuild dist matrix from scratch for trial
            dist = _init_dist_matrix()
            for (u, v) in trial:
                _update_dist_matrix(dist, u, v)
            
            if _get_max_dist(dist) <= target_diam:
                cur = trial
                improved = True
                break
    return cur

def construct():
    deadline = time.time() + 80.0
    best = None
    best_len = 10**9
    
    # Run greedy set cover with multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        g = _construct_greedy_set_cover(seed)
        if g is not None:
            # Reduce
            reduced = _try_remove_edges(g, P['diameter'])
            if len(reduced) < best_len:
                best = reduced
                best_len = len(reduced)
                
        # If we found a very good solution, we might stop early?
        # Theoretical lower bound: 
        # Path 24, diam 4.
        # Each edge can reduce diameter by at most... ?
        # If best_len <= 8, it's likely optimal or near optimal.
        if best_len <= 8:
            break

    if best is None:
        # Fallback: just return a valid solution, maybe complete bipartite minus base?
        # But that's huge. 
        # Let's just return the first greedy result if it exists, or empty if failed (should not happen)
        g = _construct_greedy_set_cover(0)
        if g:
            return [[int(u), int(v)] for (u, v) in g]
        return []

    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
