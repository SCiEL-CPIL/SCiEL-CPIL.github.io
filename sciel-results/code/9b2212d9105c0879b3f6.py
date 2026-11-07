"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute common neighbors for length-3 path feasibility
    # common[u][v] = set of w such that u-w-v is a valid path (w != u, w != v)
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cn = adj[u] & adj[v]
            cn.discard(u)
            cn.discard(v)
            common[u][v] = cn
    
    # Bron-Kerbosch for max clique (baseline)
    best_clique = []
    
    def bron_kerbosch(R, P_set, X):
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
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    # If max clique is large enough, use it directly
    if len(best_clique) >= 7:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Bipartite matching approach for length-3 subdivisions
    # For a candidate set B of k vertices, non-adjacent pairs need unique internal vertices
    # This is a bipartite matching problem: pairs -> vertices
    
    def try_length3_subdivision(B):
        """Try to find a length-3 subdivision for branch set B.
        Returns (paths, branch_list) or None."""
        k = len(B)
        B_set = set(B)
        
        # Identify non-adjacent pairs
        non_adj_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                if B[j] not in adj[B[i]]:
                    non_adj_pairs.append((i, j))
        
        # Pigeonhole: need at least as many available vertices as non-adj pairs
        if len(non_adj_pairs) > n - k:
            return None
        
        # Build assignment: each non-adj pair needs a unique internal vertex
        # pair_to_verts[pair_idx] = list of valid internal vertices
        pair_to_verts = []
        for idx, (i, j) in enumerate(non_adj_pairs):
            valid = [w for w in common[B[i]][B[j]] if w not in B_set]
            pair_to_verts.append(valid)
            if not valid:
                return None
        
        # Kuhn's algorithm for bipartite matching
        # match_vertex[v] = pair_idx assigned to vertex v, or -1
        match_vertex = {}
        
        def try_kuhn(pair_idx, seen):
            for w in pair_to_verts[pair_idx]:
                if w in seen:
                    continue
                seen.add(w)
                if w not in match_vertex or try_kuhn(match_vertex[w], seen):
                    match_vertex[w] = pair_idx
                    return True
            return False
        
        for idx in range(len(non_adj_pairs)):
            if not try_kuhn(idx, set()):
                return None
        
        # Matching found! Construct paths
        paths = []
        # Build lookup: (i,j) -> internal vertex
        pair_internal = {}
        for w, pair_idx in match_vertex.items():
            i, j = non_adj_pairs[pair_idx]
            pair_internal[(i, j)] = w
        
        for i in range(k):
            for j in range(i + 1, k):
                if B[j] in adj[B[i]]:
                    paths.append([B[i], B[j]])
                else:
                    w = pair_internal[(i, j)]
                    paths.append([B[i], w, B[j]])
        
        return paths
    
    # Try k from 12 down to 3
    best_result = None
    
    for k in range(12, 2, -1):
        if time.time() > deadline:
            break
        
        # Candidate pool: expand for smaller k to find valid sets
        # For larger k, use top-degree vertices; for smaller k, widen the pool
        if k >= 10:
            pool_size = min(n, k + 4)
        elif k >= 8:
            pool_size = min(n, k + 6)
        else:
            pool_size = min(n, k + 10)
        candidates = sorted_vertices[:pool_size]
        
        # Generate combinations with early pruning
        found = False
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            
            B = list(combo)
            # Quick check: count non-adjacent pairs and available internal vertices
            B_set = set(B)
            non_adj_count = 0
            for i in range(k):
                for j in range(i + 1, k):
                    if B[j] not in adj[B[i]]:
                        non_adj_count += 1
            if non_adj_count > n - k:
                continue
            
            result = try_length3_subdivision(B)
            if result is not None:
                best_result = (B, result)
                found = True
                break
        
        if found:
            break
        
        # If we didn't find with the current pool, try a wider pool for this k
        if k <= 9 and pool_size < n:
            wider_pool = sorted_vertices[:min(n, pool_size + 4)]
            for combo in combinations(wider_pool, k):
                if time.time() > deadline:
                    break
                B = list(combo)
                B_set = set(B)
                non_adj_count = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        if B[j] not in adj[B[i]]:
                            non_adj_count += 1
                if non_adj_count > n - k:
                    continue
                result = try_length3_subdivision(B)
                if result is not None:
                    best_result = (B, result)
                    found = True
                    break
            if found:
                break
    
    if best_result is not None:
        branches, paths = best_result
        # Ensure proper types
        branches = [int(b) for b in branches]
        paths = [[int(v) for v in p] for p in paths]
        return {'branches': branches, 'paths': paths}
    
    # Also try: if max clique is size 3-6, try to extend it with length-3 paths
    # to a larger branch set
    if best_result is None and len(best_clique) >= 3:
        # Try to add vertices to the clique as branches with length-3 paths
        clique_set = set(best_clique)
        extra_candidates = [v for v in sorted_vertices if v not in clique_set]
        
        for extra in extra_candidates:
            if time.time() > deadline:
                break
            new_B = list(best_clique) + [extra]
            new_B_set = set(new_B)
            non_adj_count = 0
            for i in range(len(new_B)):
                for j in range(i + 1, len(new_B)):
                    if new_B[j] not in adj[new_B[i]]:
                        non_adj_count += 1
            if non_adj_count > n - len(new_B):
                continue
            result = try_length3_subdivision(new_B)
            if result is not None:
                best_result = (new_B, result)
                break
    
    # Fallback: use max clique
    if best_result is None and len(best_clique) >= 3:
        branches = sorted([int(b) for b in best_clique])
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Fallback: triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    for a, b, c in combinations(range(n), 3):
        if (min(a, b), max(a, b)) in edge_set and \
           (min(a, c), max(a, c)) in edge_set and \
           (min(b, c), max(b, c)) in edge_set:
            branches = sorted([a, b, c])
            paths = [[branches[0], branches[1]], [branches[0], branches[2]], [branches[1], branches[2]]]
            return {'branches': branches, 'paths': paths}
    
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
