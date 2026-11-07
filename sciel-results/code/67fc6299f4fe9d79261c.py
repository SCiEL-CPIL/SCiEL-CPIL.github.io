"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    degrees = [len(adj[i]) for i in range(n)]
    return n, adj, adj_list, degrees

def _precompute_paths(n, adj):
    """Precompute all length-2 and length-3 paths between every pair of vertices.
    path2[u][v] = list of internal vertices w such that u-w-v is a valid path.
    path3[u][v] = list of (w1, w2) tuples such that u-w1-w2-v is a valid simple path."""
    path2 = [[[] for _ in range(n)] for _ in range(n)]
    path3 = [[[] for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v or v in adj[u]:
                continue
            # Length 2
            common = adj[u] & adj[v]
            if common:
                path2[u][v] = list(common)
                path2[v][u] = list(common)
            # Length 3
            cands = []
            for w1 in adj[u]:
                if w1 == v:
                    continue
                for w2 in adj[v]:
                    if w2 == u or w2 == w1:
                        continue
                    if w1 in adj[w2]:
                        cands.append((w1, w2))
            if cands:
                path3[u][v] = cands
                path3[v][u] = [(w2, w1) for (w1, w2) in cands]
    return path2, path3

def _kuhn_match(pairs_to_verts, n):
    """Bipartite matching: pairs (left) to internal vertices (right).
    pairs_to_verts[i] = list of valid internal vertices for pair i.
    Returns dict pair_idx -> vertex, or None if no perfect matching."""
    m = len(pairs_to_verts)
    if m == 0:
        return {}
    match_v = [-1] * n  # vertex -> pair_idx
    def try_kuhn(pi, seen):
        for v in pairs_to_verts[pi]:
            if seen[v]:
                continue
            seen[v] = True
            if match_v[v] == -1 or try_kuhn(match_v[v], seen):
                match_v[v] = pi
                return True
        return False
    for i in range(m):
        seen = [False] * n
        if not try_kuhn(i, seen):
            return None
    result = {}
    for v in range(n):
        if match_v[v] != -1:
            result[match_v[v]] = v
    return result

def _try_subdivision(B, n, adj, path2, path3):
    """Try to find a subdivision for branch set B.
    Returns list of paths (each a list of vertices), or None."""
    B_set = set(B)
    k = len(B)
    if k < 2:
        return None
    pairs = list(combinations(B, 2))
    adj_pairs = []
    non_adj_pairs = []
    for u, v in pairs:
        if v in adj[u]:
            adj_pairs.append((u, v))
        else:
            non_adj_pairs.append((u, v))
    
    # Quick feasibility: need at least len(non_adj_pairs) internal vertices
    available = n - k
    if len(non_adj_pairs) > available:
        return None
    
    # Try pure length-2 matching
    if non_adj_pairs:
        pairs_to_verts = []
        for u, v in non_adj_pairs:
            valid = [w for w in path2[u][v] if w not in B_set]
            pairs_to_verts.append(valid)
            if not valid:
                break
        else:
            match = _kuhn_match(pairs_to_verts, n)
            if match is not None:
                paths = []
                for u, v in adj_pairs:
                    paths.append([u, v])
                for idx, (u, v) in enumerate(non_adj_pairs):
                    w = match[idx]
                    paths.append([u, w, v])
                return paths
    
    # Try mixed: use backtracking to assign length-2 or length-3 paths
    # Sort non-adj pairs by number of options (most constrained first)
    def options_count(u, v):
        cnt = len([w for w in path2[u][v] if w not in B_set])
        cnt += len([(w1, w2) for (w1, w2) in path3[u][v] if w1 not in B_set and w2 not in B_set])
        return cnt
    
    non_adj_sorted = sorted(non_adj_pairs, key=lambda p: options_count(p[0], p[1]))
    
    used_int = set()
    assignment = {}  # (u,v) -> path
    
    def backtrack(idx):
        if idx == len(non_adj_sorted):
            return True
        u, v = non_adj_sorted[idx]
        # Try length-2 options first (cheaper)
        for w in path2[u][v]:
            if w in B_set or w in used_int:
                continue
            used_int.add(w)
            assignment[(u, v)] = [u, w, v]
            if backtrack(idx + 1):
                return True
            used_int.remove(w)
            del assignment[(u, v)]
        # Try length-3 options
        for w1, w2 in path3[u][v]:
            if w1 in B_set or w2 in B_set:
                continue
            if w1 in used_int or w2 in used_int:
                continue
            used_int.add(w1)
            used_int.add(w2)
            assignment[(u, v)] = [u, w1, w2, v]
            if backtrack(idx + 1):
                return True
            used_int.remove(w1)
            used_int.remove(w2)
            del assignment[(u, v)]
        return False
    
    if backtrack(0):
        paths = []
        for u, v in adj_pairs:
            paths.append([u, v])
        for u, v in non_adj_pairs:
            paths.append(assignment[(u, v)])
        return paths
    
    return None

def _bron_kerbosch(n, adj):
    """Find maximum clique."""
    best_clique = []
    def bk(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bk(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    bk(set(), set(range(n)), set())
    return best_clique

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n, adj, adj_list, degrees = _build_graph()
    path2, path3 = _precompute_paths(n, adj)

    if time.time() > deadline:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best_k = 0
    best_branches = None
    best_paths = None

    def update_best(B, paths):
        nonlocal best_k, best_branches, best_paths
        if paths is not None and len(B) > best_k:
            best_k = len(B)
            best_branches = list(B)
            best_paths = paths

    # Phase 1: Maximum clique as baseline
    max_clique = _bron_kerbosch(n, adj)
    if len(max_clique) >= 2:
        B = list(max_clique)
        paths = []
        ok = True
        for u, v in combinations(B, 2):
            if v in adj[u]:
                paths.append([u, v])
            else:
                ok = False
                break
        if ok:
            update_best(B, paths)

    # Phase 2: Systematic search for larger subdivisions
    # Strategy: build candidate branch sets by including high-degree vertices
    # and testing with the fast matching validator.
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    # Try increasing sizes from max_clique+1 upward
    max_k_to_try = min(n, 14)
    found = False
    for k in range(best_k + 1, max_k_to_try + 1):
        if time.time() > deadline:
            break
        if found:
            break
        
        # Generate candidate branch sets
        # Pool: top (k + 10) by degree
        pool_size = min(k + 10, n)
        base_pool = sorted_vertices[:pool_size]
        
        # For small k, enumerate all combinations from a moderate pool
        if k <= 10:
            pool_for_enum = sorted_vertices[:min(k + 6, n)]
            combos = list(combinations(pool_for_enum, k))
        else:
            # For larger k, sample from the pool
            random.seed(42 + k)
            combos = []
            seen_sets = set()
            for _ in range(3000):
                if time.time() > deadline:
                    break
                cand = tuple(sorted(random.sample(base_pool, k)))
                if cand not in seen_sets:
                    seen_sets.add(cand)
                    combos.append(cand)
        
        # Also add some random candidates from all vertices
        random.seed(123 + k)
        for _ in range(2000):
            if time.time() > deadline:
                break
            cand = tuple(sorted(random.sample(range(n), k)))
            if cand not in seen_sets:
                seen_sets.add(cand)
                combos.append(cand)
        
        for B_tuple in combos:
            if time.time() > deadline:
                break
            B = list(B_tuple)
            paths = _try_subdivision(B, n, adj, path2, path3)
            if paths is not None:
                update_best(B, paths)
                found = True
                break
        if found:
            # Try to go one more
            continue
        else:
            # Didn't find any of this size, but keep trying larger
            # (unlikely but possible with different vertex sets)
            pass

    # Phase 3: Local improvement - try to add one more vertex to best set
    if best_branches is not None:
        current_B = set(best_branches)
        for extra in range(n):
            if time.time() > deadline:
                break
            if extra in current_B:
                continue
            candidate = list(best_branches) + [extra]
            paths = _try_subdivision(candidate, n, adj, path2, path3)
            if paths is not None:
                update_best(candidate, paths)
                # Try to add another
                current_B.add(extra)
    
    # Phase 4: Random restarts with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if time.time() > deadline:
            break
        random.seed(seed)
        for k in range(min(best_k + 3, max_k_to_try + 1), 1, -1):
            if time.time() > deadline:
                break
            for _ in range(500):
                if time.time() > deadline:
                    break
                cand = tuple(sorted(random.sample(range(n), k)))
                B = list(cand)
                paths = _try_subdivision(B, n, adj, path2, path3)
                if paths is not None:
                    update_best(B, paths)
                    break

    # Fallback
    if best_branches is None or best_k < 2:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    # Convert to proper format
    branches_out = [int(b) for b in best_branches]
    paths_out = []
    for path in best_paths:
        paths_out.append([int(x) for x in path])

    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
