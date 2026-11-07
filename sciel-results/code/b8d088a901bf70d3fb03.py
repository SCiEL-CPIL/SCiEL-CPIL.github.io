"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    degree = [len(adj[v]) for v in range(n)]

    # ---- Bron-Kerbosch to find maximum clique ----
    max_clique = []

    def bronk(R, P_set, X_set):
        nonlocal max_clique
        if time.time() > deadline - 5:
            return
        if not P_set and not X_set:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        U = P_set | X_set
        if not U:
            return
        pivot = None
        max_inter = -1
        for u in U:
            inter = len(P_set & adj[u])
            if inter > max_inter:
                max_inter = inter
                pivot = u
        candidates = list(P_set - adj[pivot])
        for v in candidates:
            bronk(R | {v}, P_set & adj[v], X_set & adj[v])
            P_set.discard(v)
            X_set.add(v)
            if time.time() > deadline - 5:
                return

    all_vertices = set(range(n))
    bronk(set(), all_vertices, set())

    if len(max_clique) < 2:
        max_clique = [edges[0][0], edges[0][1]]

    # ---- Path finding with backtracking and multiple candidate paths ----
    def get_candidate_paths(src, dst, forbidden, used, max_len=6, limit=10):
        """Find multiple candidate paths from src to dst avoiding forbidden and used."""
        if src == dst:
            return [[src]]
        blocked = forbidden | used
        paths = []
        visited = {src}
        queue = deque([(src, [src])])
        
        while queue:
            node, path = queue.popleft()
            if len(path) > max_len:
                continue
            if node == dst:
                paths.append(path)
                if len(paths) >= limit:
                    break
                continue
            for nb in adj[node]:
                if nb in blocked or nb in visited:
                    continue
                visited.add(nb)
                queue.append((nb, path + [nb]))
        
        paths.sort(key=len)
        return paths[:limit]

    def find_paths_with_backtracking(branches):
        """Find vertex-disjoint interior paths for all pairs using backtracking.
        Returns list of paths or None."""
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(range(k), 2))
        branch_set = set(branches)
        num_pairs = len(pairs)
        
        # Order pairs by difficulty
        def pair_key(idx):
            i, j = pairs[idx]
            bi, bj = branches[i], branches[j]
            if (bi, bj) in edge_set:
                return 0
            common = (adj[bi] & adj[bj]) - branch_set
            return len(common)
        
        order = sorted(range(num_pairs), key=pair_key)
        assignment = [None] * num_pairs
        
        def backtrack(pos, used_int):
            if time.time() > deadline - 2:
                return False
            if pos == num_pairs:
                return True
            
            pair_idx = order[pos]
            i, j = pairs[pair_idx]
            bi, bj = branches[i], branches[j]
            forbidden = branch_set - {bi, bj}
            
            if (bi, bj) in edge_set:
                assignment[pair_idx] = [bi, bj]
                if backtrack(pos + 1, used_int):
                    return True
                assignment[pair_idx] = None
                return False
            
            candidates = get_candidate_paths(bi, bj, forbidden, used_int)
            for path in candidates:
                interior = set(path[1:-1])
                if interior & used_int:
                    continue
                assignment[pair_idx] = path
                used_int_new = used_int | interior
                if backtrack(pos + 1, used_int_new):
                    return True
                assignment[pair_idx] = None
            
            return False
        
        if backtrack(0, set()):
            return [assignment[idx] for idx in range(num_pairs)]
        return None

    # ---- Max-SAT style local search for branch selection ----
    def max_sat_local_search(initial_branches, candidate_pool, max_iters=200):
        """Local search to maximize number of routable branches.
        Flips variables (add/remove vertices) to minimize constraint violations."""
        current = list(initial_branches)
        current_paths = find_paths_with_backtracking(current)
        if current_paths is None:
            return None, None
        
        best = (list(current), current_paths)
        best_size = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline - 8:
                break
            
            improved = False
            
            # Try adding a vertex
            for v in candidate_pool:
                if v in current:
                    continue
                if time.time() > deadline - 4:
                    break
                new_branches = current + [v]
                new_paths = find_paths_with_backtracking(new_branches)
                if new_paths is not None:
                    current = new_branches
                    current_paths = new_paths
                    best = (list(current), current_paths)
                    best_size = len(current)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing a vertex and re-adding a different one
            for idx, v in enumerate(list(current)):
                if time.time() > deadline - 4:
                    break
                reduced = [x for i, x in enumerate(current) if i != idx]
                if len(reduced) < 2:
                    continue
                reduced_paths = find_paths_with_backtracking(reduced)
                if reduced_paths is None:
                    continue
                
                # Try to add a different vertex to the reduced set
                for w in candidate_pool:
                    if w in reduced:
                        continue
                    if time.time() > deadline - 3:
                        break
                    new_branches = reduced + [w]
                    new_paths = find_paths_with_backtracking(new_branches)
                    if new_paths is not None and len(new_branches) > best_size:
                        current = new_branches
                        current_paths = new_paths
                        best = (list(current), current_paths)
                        best_size = len(current)
                        break
        
        return best

    # ---- Main search ----
    candidate_pool = sorted(range(n), key=lambda v: degree[v], reverse=True)
    
    best_branches = None
    best_paths = None
    best_size = 0
    
    # Strategy 1: Start from max clique and extend
    if max_clique:
        result = max_sat_local_search(max_clique, candidate_pool)
        if result[0] is not None and len(result[0]) > best_size:
            best_branches, best_paths = result
            best_size = len(result[0])
    
    # Strategy 2: Start from top-degree pairs and extend
    top_deg = sorted(range(n), key=lambda v: degree[v], reverse=True)
    for seed_idx in range(min(8, len(top_deg))):
        if time.time() > deadline - 15:
            break
        seed_v = top_deg[seed_idx]
        seed = [seed_v]
        for w in top_deg:
            if w == seed_v:
                continue
            if (seed_v, w) in edge_set:
                seed.append(w)
                break
        if len(seed) < 2:
            continue
        
        result = max_sat_local_search(seed, candidate_pool)
        if result[0] is not None and len(result[0]) > best_size:
            best_branches, best_paths = result
            best_size = len(result[0])
    
    # Strategy 3: Random restarts
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        random_seed = random.sample(range(n), min(3, n))
        if len(random_seed) < 2:
            continue
        result = max_sat_local_search(random_seed, candidate_pool)
        if result[0] is not None and len(result[0]) > best_size:
            best_branches, best_paths = result
            best_size = len(result[0])
    
    # Fallback: two connected vertices
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    elif best_paths is None:
        # Try one more time
        best_paths = find_paths_with_backtracking(best_branches)
        if best_paths is None:
            u, v = edges[0]
            best_branches = [int(u), int(v)]
            best_paths = [[int(u), int(v)]]
    
    # Ensure types
    branches_out = [int(b) for b in best_branches]
    paths_out = []
    for p in best_paths:
        paths_out.append([int(x) for x in p])
    
    # Final sanity check
    if len(branches_out) < 2:
        u, v = edges[0]
        branches_out = [int(u), int(v)]
        paths_out = [[int(u), int(v)]]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
