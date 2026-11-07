import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build a flow network to check if a given branch set admits internally-disjoint paths
    # for all pairs. We model it as: each non-branch vertex v is split into v_in -> v_out with
    # capacity 1. Branch vertices are not split (infinite capacity). Edges go from u_out (or u
    # if branch) to v_in (or v if branch) with infinite capacity.
    # For each pair (u,v) of branches, we need to route 1 unit from u to v.
    # This is a multi-commodity flow, which is NP-hard in general. However, for small branch
    # sets (k <= 8 or so), we can use a different approach: iteratively find paths with a
    # constraint that they don't share interior vertices.
    
    # Alternative approach: use a SAT-like or backtracking approach for path finding with
    # the disjointness constraint. For k branches, we need C(k,2) paths.
    
    # Better approach: use max-flow for each pair sequentially but with a "used" set,
    # and try different orderings. The key insight is that the order of path assignment
    # matters. We can use a randomized local search.
    
    # Most robust approach for small k: use a backtracking search over paths.
    # For k=4, we have 6 paths. For k=5, 10 paths. This is manageable with good pruning.
    
    # Let's use a different structural approach:
    # 1. Find a good initial branch set (clique or greedy).
    # 2. Use a local search that tries to add/remove/swap vertices.
    # 3. For feasibility check, use a randomized path-finding with multiple restarts.
    
    def find_path_bfs(src, dst, forbidden):
        """Find a simple path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return None
        if dst in forbidden:
            return None
        queue = deque()
        queue.append((src, [src]))
        visited = {src}
        while queue:
            node, path = queue.popleft()
            for nb in adj[node]:
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def check_feasible(branches, max_restarts=50):
        """Check if branches admit internally-disjoint paths for all pairs.
        Returns paths dict or None."""
        import random
        branch_set = set(branches)
        k = len(branches)
        if k < 2:
            return None
        
        # Try multiple random orderings of pairs
        pairs = list(combinations(branches, 2))
        
        for restart in range(max_restarts):
            random.shuffle(pairs)
            paths = {}
            used_interior = set()
            success = True
            for u, v in pairs:
                forbidden = (branch_set - {u, v}) | used_interior
                path = find_path_bfs(u, v, forbidden)
                if path is None:
                    success = False
                    break
                for vertex in path[1:-1]:
                    used_interior.add(vertex)
                paths[(u, v)] = path
            if success:
                return paths
        return None
    
    # Find maximum clique using Bron-Kerbosch
    best_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}
    
    bron_kerbosch(set(), set(range(n)), set())
    
    best_count = 0
    best_result = None
    
    def try_branch_set(branches):
        nonlocal best_count, best_result
        if time.time() > deadline:
            return
        if len(branches) < 2:
            return
        paths = check_feasible(branches)
        if paths is not None:
            if len(branches) > best_count:
                best_count = len(branches)
                result_paths = []
                for u, v in combinations(sorted(branches), 2):
                    result_paths.append([int(x) for x in paths[(u, v)]])
                best_result = {'branches': [int(x) for x in sorted(branches)], 'paths': result_paths}
    
    # Try the maximum clique
    if len(best_clique) >= 2:
        try_branch_set(sorted(best_clique))
    
    # Greedy expansion: start from clique, try adding vertices
    if len(best_clique) >= 2:
        current = sorted(best_clique)
        try_branch_set(current)
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = False
            for candidate in range(n):
                if candidate in current:
                    continue
                new_set = current + [candidate]
                paths = check_feasible(new_set)
                if paths is not None:
                    current = new_set
                    try_branch_set(current)
                    improved = True
                    break
            if not improved:
                break
    
    # Local search: random restarts with hill climbing
    import random
    random.seed(123)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with a random set
        size = random.randint(3, min(n, 7))
        current = random.sample(range(n), size)
        try_branch_set(current)
        
        # Hill climbing: try to add a vertex
        for _ in range(20):
            if time.time() > deadline:
                break
            if len(current) >= n:
                break
            candidates = [v for v in range(n) if v not in current]
            random.shuffle(candidates)
            improved = False
            for candidate in candidates:
                new_set = current + [candidate]
                paths = check_feasible(new_set)
                if paths is not None:
                    current = new_set
                    try_branch_set(current)
                    improved = True
                    break
            if not improved:
                break
        
        # Try swaps
        if time.time() > deadline:
            break
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(current) < 2:
                break
            i, j = random.sample(range(len(current)), 2)
            old = current[i]
            candidates = [v for v in range(n) if v not in current]
            random.shuffle(candidates)
            improved = False
            for candidate in candidates:
                new_set = current[:i] + [candidate] + current[i+1:]
                paths = check_feasible(new_set)
                if paths is not None:
                    current = new_set
                    try_branch_set(current)
                    improved = True
                    break
            if not improved:
                break
    
    # Try all subsets of size up to 5 from a candidate pool
    # Candidate pool: vertices with high degree
    degree_sorted = sorted(range(n), key=lambda v: -len(adj[v]))
    pool = degree_sorted[:15]
    
    for size in range(4, min(n, 7)):
        if time.time() > deadline:
            break
        for subset in combinations(pool, size):
            if time.time() > deadline:
                break
            try_branch_set(list(subset))
    
    if best_result is not None:
        return best_result
    
    # Fallback
    for u, v in P['edges']:
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}
