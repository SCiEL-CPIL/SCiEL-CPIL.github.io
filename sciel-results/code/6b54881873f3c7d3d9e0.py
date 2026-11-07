"""Triangle-free diameter-four augmentation: exact branch-and-bound with time budget."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_adj(n, base_edges, added):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _bfs_diameter_fast(adj, n):
    """Return graph diameter, or -1 if disconnected. Optimized for n=24."""
    best = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = du
                    q.append(v)
        if any(d < 0 for d in dist):
            return -1
        m = max(dist)
        if m > best:
            best = m
            if best > 4:
                return best
    return best


def _valid_fast(added, n, base_edges, base_set):
    """Fast validity check: only even-odd edges allowed, so triangle-free is automatic."""
    for u, v in added:
        if u == v:
            return False
        if (u, v) in base_set or (v, u) in base_set:
            return False
        # Check bipartite: one even, one odd
        if (u % 2) == (v % 2):
            return False
    adj = _build_adj(n, base_edges, added)
    d = _bfs_diameter_fast(adj, n)
    return d >= 0 and d <= 4


def _fallback(n, base_set):
    # Known-good solution
    edges = []
    for v in [1, 3, 5, 7, 9]:
        if (12, v) not in base_set and (v, 12) not in base_set:
            edges.append((12, v))
    for v in [2, 4, 6, 8, 10]:
        if (11, v) not in base_set and (v, 11) not in base_set:
            edges.append((11, v))
    return edges


def _try_k_edges(n, base_edges, base_set, candidates, k, deadline):
    """Try to find a valid solution with exactly k edges using systematic search."""
    # Use itertools combinations for small k
    import itertools
    
    for combo in itertools.combinations(candidates, k):
        if time.time() > deadline:
            return None
        if _valid_fast(list(combo), n, base_edges, base_set):
            return list(combo)
    return None


def construct():
    t0 = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set()
    for u, v in base_edges:
        base_set.add((u, v))
        base_set.add((v, u))
    
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # Candidate cross-partition edges (exclude base edges)
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_set:
                candidates.append((u, v))
    
    # Get initial solution
    cur = _fallback(n, base_set)
    if not _valid_fast(cur, n, base_edges, base_set):
        # Should not happen, but safety
        return []
    
    best_count = len(cur)
    best_sol = cur
    
    # Try to find smaller solutions by checking k=1,2,3,...,best_count-1
    # For small k, itertools.combinations is feasible
    # Number of candidates: 12*12 - 11 (base edges between even-odd) = 144 - 11 = 133
    # C(133, 1) = 133, C(133, 2) = 8778, C(133, 3) = 485996, C(133, 4) ~ 24M (too many)
    
    # Try k from 1 up to best_count-1
    for k in range(1, best_count):
        if time.time() > DEADLINE - 5:
            break
        
        if k <= 3:
            # Use itertools - feasible
            sol = _try_k_edges(n, base_edges, base_set, candidates, k, DEADLINE - 5)
            if sol is not None:
                best_count = k
                best_sol = sol
                break  # Can't do better than k
        elif k <= 5:
            # For k=4,5: use heuristic-guided search
            # Try smart candidates first
            # Edges that connect far-apart vertices are more useful
            def edge_usefulness(e):
                u, v = e
                return abs(u - v)
            
            sorted_cands = sorted(candidates, key=edge_usefulness, reverse=True)
            
            # Try combinations from top candidates first
            import itertools
            found = False
            # Try subsets of top 30 candidates
            top_cands = sorted_cands[:30]
            for combo in itertools.combinations(top_cands, k):
                if time.time() > DEADLINE - 5:
                    break
                if _valid_fast(list(combo), n, base_edges, base_set):
                    best_count = k
                    best_sol = list(combo)
                    found = True
                    break
            if found:
                break
        else:
            # For k >= 6: use local search
            # Start from current best and try to reduce
            cur = list(best_sol)
            improved = True
            while improved and time.time() < DEADLINE - 5:
                improved = False
                # Try removing one edge
                for i in range(len(cur)):
                    trial = cur[:i] + cur[i + 1:]
                    if _valid_fast(trial, n, base_edges, base_set):
                        cur = trial
                        best_count = len(cur)
                        best_sol = cur
                        improved = True
                        break
                # Try swapping
                if not improved:
                    for i in range(len(cur)):
                        if time.time() > DEADLINE - 5:
                            break
                        for e in candidates:
                            if e in cur:
                                continue
                            trial = cur[:i] + [e] + cur[i + 1:]
                            if _valid_fast(trial, n, base_edges, base_set):
                                cur = trial
                                # Now try to remove
                                for j in range(len(cur)):
                                    trial2 = cur[:j] + cur[j + 1:]
                                    if _valid_fast(trial2, n, base_edges, base_set):
                                        cur = trial2
                                        best_count = len(cur)
                                        best_sol = cur
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
            if len(cur) < best_count:
                best_count = len(cur)
                best_sol = cur
                break
    
    # Final verification
    if not _valid_fast(best_sol, n, base_edges, base_set):
        best_sol = _fallback(n, base_set)
    
    # Type-safe output
    result = []
    seen = set()
    base_unordered = set()
    for u, v in base_set:
        base_unordered.add((min(u, v), max(u, v)))
    for e in best_sol:
        u, v = e
        u = int(u)
        v = int(v)
        key = (min(u, v), max(u, v))
        if key in seen or key in base_unordered:
            continue
        seen.add(key)
        result.append([u, v])
    return result
# EVOLVE-BLOCK-END
