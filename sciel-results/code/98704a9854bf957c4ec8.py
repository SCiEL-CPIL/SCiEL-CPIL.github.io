import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Build a fast adjacency check
    adj_bool = [[False]*n for _ in range(n)]
    for u, v in P['edges']:
        adj_bool[u][v] = True
        adj_bool[v][u] = True

    def time_ok():
        return time.time() < deadline

    # Find all maximal cliques using Bron-Kerbosch with pivot
    def find_maximal_cliques():
        cliques = []
        R = set()
        Pset = set(range(n))
        X = set()
        def bronk(R, Pset, X):
            if not Pset and not X:
                cliques.append(frozenset(R))
                return
            # Pivot selection
            if Pset or X:
                U = Pset | X
                pivot = max(U, key=lambda v: len(Pset & adj[v]))
                candidates = Pset - adj[pivot]
            else:
                candidates = Pset
            for v in list(candidates):
                if not time_ok():
                    return
                bronk(R | {v}, Pset & adj[v], X & adj[v])
                Pset.remove(v)
                X.add(v)
        bronk(R, Pset, X)
        return cliques

    # Try to find internally disjoint paths for a given set of branches
    # Using a backtracking approach with good ordering heuristics
    def try_branch_set(branches):
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Order pairs by difficulty: prefer pairs that are "closer" first
        # or pairs that have more options
        def pair_difficulty(pair):
            a, b = pair
            # Count common non-branch neighbors (potential interior vertices)
            common = adj[a] & adj[b] - branch_set
            # Also consider distance
            return -len(common)  # More common neighbors = easier
        
        # Sort pairs: easier first (more options)
        pairs_sorted = sorted(pairs, key=pair_difficulty)
        
        def find_paths(remaining_pairs, used_int, current_paths):
            if not time_ok():
                return None
            if not remaining_pairs:
                return current_paths
            
            # Heuristic: try the "hardest" remaining pair first (fewest options)
            # This is a "most constrained variable" heuristic
            best_idx = 0
            best_score = float('inf')
            for idx, (a, b) in enumerate(remaining_pairs):
                forbidden = used_int | (branch_set - {a, b})
                # Count available interior vertices
                avail = (adj[a] | adj[b]) - forbidden - {a, b}
                score = len(avail)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            a, b = remaining_pairs[best_idx]
            rest = remaining_pairs[:best_idx] + remaining_pairs[best_idx+1:]
            forbidden = used_int | (branch_set - {a, b})
            
            # Find all possible short paths (up to some length limit)
            paths = find_short_paths(a, b, forbidden, adj, max_len=6)
            
            # Try each path
            for path in paths:
                interior = set(path[1:-1])
                if interior & used_int:
                    continue
                new_used = used_int | interior
                result = find_paths(rest, new_used, current_paths + [path])
                if result is not None:
                    return result
            return None
        
        result = find_paths(pairs_sorted, set(), [])
        if result is not None:
            return result
        
        # Try different orderings
        import random
        rng = random.Random(42)
        for attempt in range(20):
            if not time_ok():
                break
            shuffled = pairs_sorted[:]
            rng.shuffle(shuffled)
            result = find_paths(shuffled, set(), [])
            if result is not None:
                return result
        return None

    def find_short_paths(start, end, forbidden, adj, max_len=6):
        """Find all simple paths from start to end with length <= max_len, avoiding forbidden."""
        if start == end:
            return [[start]]
        paths = []
        # DFS with length limit
        def dfs(node, path, visited):
            if len(path) - 1 > max_len:
                return
            if node == end:
                paths.append(path[:])
                return
            for nb in adj[node]:
                if nb in visited:
                    continue
                if nb in forbidden and nb != end:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
        
        dfs(start, [start], {start})
        # Sort by length
        paths.sort(key=len)
        return paths

    # Main search strategy:
    # 1. Find all maximal cliques
    # 2. Try each clique (larger first)
    # 3. Try to expand cliques by adding vertices
    # 4. Try random subsets of various sizes

    cliques = find_maximal_cliques()
    
    # Sort cliques by size descending
    cliques = sorted(cliques, key=lambda c: -len(c))
    
    best_branches = None
    best_paths = None
    
    # Try each clique
    for clique in cliques:
        if not time_ok():
            break
        clique_list = list(clique)
        if best_branches is not None and len(clique_list) <= len(best_branches):
            continue
        paths_result = try_branch_set(clique_list)
        if paths_result is not None:
            best_branches = clique_list
            best_paths = paths_result
    
    # Try to expand the best branch set
    if best_branches is not None:
        current = best_branches[:]
        expanded = True
        while expanded and time_ok():
            expanded = False
            current_set = set(current)
            candidates = [v for v in range(n) if v not in current_set]
            # Sort by degree descending
            candidates.sort(key=lambda v: -len(adj[v]))
            for v in candidates:
                if not time_ok():
                    break
                new_branches = current + [v]
                paths_result = try_branch_set(new_branches)
                if paths_result is not None:
                    current = new_branches
                    best_branches = current[:]
                    best_paths = paths_result
                    expanded = True
                    break
    
    # Try random subsets
    import random
    rng2 = random.Random(999)
    for size in range(n, 1, -1):
        if not time_ok():
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(50):
            if not time_ok():
                break
            random_set = rng2.sample(range(n), size)
            paths_result = try_branch_set(random_set)
            if paths_result is not None:
                best_branches = random_set
                best_paths = paths_result
                break
    
    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]
    
    return {'branches': best_branches, 'paths': best_paths}
