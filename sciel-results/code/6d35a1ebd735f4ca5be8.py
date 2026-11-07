"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

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

    # ---- Path finding with backtracking ----
    def find_paths_with_backtracking(branches, max_depth=3):
        """Find vertex-disjoint interior paths for all pairs using backtracking.
        Returns list of paths or None."""
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(range(k), 2))
        branch_set = set(branches)
        num_pairs = len(pairs)
        
        # Pre-compute: for each pair, find all candidate paths up to some length
        # Use BFS to find shortest path, then try to find alternatives if needed
        
        def get_candidate_paths(src, dst, forbidden, max_len=6):
            """Find multiple candidate paths from src to dst avoiding forbidden."""
            paths = []
            if src == dst:
                return [[src]]
            blocked = forbidden
            visited = {src}
            queue = deque([(src, [src])])
            found_shortest = False
            shortest_len = float('inf')
            
            while queue:
                node, path = queue.popleft()
                if len(path) > max_len:
                    continue
                if node == dst:
                    paths.append(path)
                    if not found_shortest:
                        found_shortest = True
                        shortest_len = len(path)
                    continue
                for nb in adj[node]:
                    if nb in blocked or nb in visited:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
            
            # Sort by length
            paths.sort(key=len)
            return paths[:10]  # Keep top 10 candidates
        
        # State: assignment of paths to pairs
        assignment = [None] * num_pairs
        used_interior = set()
        
        # Order pairs by difficulty (fewest common neighbors first)
        def pair_key(idx):
            i, j = pairs[idx]
            bi, bj = branches[i], branches[j]
            if (bi, bj) in edge_set:
                return 0
            common = (adj[bi] & adj[bj]) - branch_set
            return len(common)
        
        order = sorted(range(num_pairs), key=pair_key)
        
        def backtrack(pos, used_int):
            if time.time() > deadline - 3:
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
            
            candidates = get_candidate_paths(bi, bj, forbidden)
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
            result = []
            for idx in range(num_pairs):
                result.append(assignment[idx])
            return result
        return None

    # ---- Greedy extension with multiple strategies ----
    def try_extend(start_branches, candidate_pool, strategy='greedy'):
        """Try to extend the branch set. Returns (branches, paths) or None."""
        current = list(start_branches)
        paths = find_paths_with_backtracking(current)
        if paths is None:
            return None
        
        best = (list(current), paths)
        
        extended = True
        while extended:
            extended = False
            if time.time() > deadline - 10:
                break
            
            if strategy == 'greedy':
                candidates = [v for v in candidate_pool if v not in current]
            elif strategy == 'random':
                candidates = [v for v in candidate_pool if v not in current]
                random.shuffle(candidates)
            else:
                candidates = [v for v in candidate_pool if v not in current]
            
            for v in candidates:
                if time.time() > deadline - 5:
                    break
                new_branches = current + [v]
                new_paths = find_paths_with_backtracking(new_branches)
                if new_paths is not None:
                    current = new_branches
                    best = (list(current), new_paths)
                    extended = True
                    break
        
        return best

    # ---- Main search ----
    best_branches = list(max_clique)
    best_paths = find_paths_with_backtracking(best_branches)
    best_size = len(best_branches)
    
    if best_paths is None:
        # Fallback: direct edges only
        best_paths = []
        for u, v in combinations(best_branches, 2):
            best_paths.append([u, v])

    # Candidate pool: all vertices sorted by degree
    candidate_pool = sorted(range(n), key=lambda v: degree[v], reverse=True)
    
    # Strategy 1: Greedy extension from max clique
    result = try_extend(best_branches, candidate_pool, 'greedy')
    if result and len(result[0]) > best_size:
        best_branches, best_paths = result
        best_size = len(best_branches)
    
    # Strategy 2: Multiple random restarts
    for restart in range(15):
        if time.time() > deadline - 15:
            break
        
        # Random seed: pick 2-3 random high-degree vertices
        seed_size = random.randint(2, 3)
        seed_vertices = random.sample(candidate_pool[:10], min(seed_size, 10))
        seed_branches = list(seed_vertices)
        
        # Ensure at least 2 and they form a valid starting point
        if len(seed_branches) < 2:
            continue
        
        result = try_extend(seed_branches, candidate_pool, 'random')
        if result and len(result[0]) > best_size:
            best_branches, best_paths = result
            best_size = len(best_branches)
    
    # Strategy 3: Local search - try swapping a branch vertex
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Try to swap one branch for a non-branch
            cur_branches = list(best_branches)
            improved = False
            
            for bi in range(len(cur_branches)):
                if time.time() > deadline - 8:
                    break
                old_b = cur_branches[bi]
                for v in candidate_pool:
                    if v in cur_branches:
                        continue
                    new_branches = cur_branches[:bi] + [v] + cur_branches[bi+1:]
                    new_paths = find_paths_with_backtracking(new_branches)
                    if new_paths is not None:
                        # Check if this is actually an improvement (same size but maybe better structure)
                        # For now, just accept if it works
                        best_branches = new_branches
                        best_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # ---- Final verification and type conversion ----
    if best_paths is None or len(best_paths) != len(list(combinations(best_branches, 2))):
        # Ultimate fallback: two connected vertices
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
