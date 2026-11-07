import time
import itertools
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def time_up():
        return time.time() > deadline
    
    def find_disjoint_paths(branches, forbidden):
        """
        Try to find one simple path for each unordered pair of branches,
        where all path interiors are pairwise disjoint and avoid all branches and forbidden vertices.
        Returns dict mapping (i,j) -> path, or None if impossible.
        """
        k = len(branches)
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((i, j))
        
        # Order pairs by difficulty: fewer available routes first
        # For each pair, count how many common neighbors or nearby paths exist
        def pair_difficulty(pair):
            i, j = pair
            u, v = branches[i], branches[j]
            # If directly adjacent, easy
            if v in adj[u]:
                return 0
            # Count vertices adjacent to both (excluding branches and forbidden)
            common = (adj[u] & adj[v]) - set(branches) - forbidden
            return -len(common)  # Fewer common neighbors = harder
        
        pairs.sort(key=pair_difficulty)
        
        paths = {}
        used_interiors = set()
        
        for (i, j) in pairs:
            u, v = branches[i], branches[j]
            # Find a simple path from u to v avoiding all branches (except u,v) and used_interiors
            # BFS with constraints
            forbidden_set = set(branches) - {u, v}
            forbidden_set |= used_interiors
            
            # BFS
            from collections import deque
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue and not time_up():
                cur = queue.popleft()
                if cur == v:
                    found = True
                    break
                for nb in sorted(adj[cur]):
                    if nb not in parent and nb not in forbidden_set:
                        parent[nb] = cur
                        queue.append(nb)
            
            if not found:
                return None
            
            # Reconstruct path
            path = []
            cur = v
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            
            # Verify path
            if path[0] != u or path[-1] != v:
                return None
            # Check interiors don't conflict
            interior = set(path[1:-1])
            if interior & (forbidden_set | set(branches)):
                return None
            
            paths[(i, j)] = path
            used_interiors |= interior
        
        return paths
    
    def try_greedy_growth(initial_branches, max_branches=None):
        """
        Start with initial_branches, greedily add vertices to maximize subdivision size.
        Returns (branches, paths) if successful, else None.
        """
        if max_branches is None:
            max_branches = n
        
        branches = list(initial_branches)
        if len(branches) < 2:
            return None
        
        # Verify initial set has disjoint paths
        paths = find_disjoint_paths(branches, set())
        if paths is None:
            return None
        
        # Try to add more vertices
        candidates = [v for v in range(n) if v not in set(branches)]
        
        while len(branches) < max_branches and not time_up():
            # Try each candidate
            added = False
            # Sort candidates by heuristic: prefer vertices adjacent to many existing branches
            candidates.sort(key=lambda v: -len(adj[v] & set(branches)))
            
            for cand in candidates:
                if time_up():
                    break
                new_branches = branches + [cand]
                new_paths = find_disjoint_paths(new_branches, set())
                if new_paths is not None:
                    branches = new_branches
                    paths = new_paths
                    candidates = [v for v in range(n) if v not in set(branches)]
                    added = True
                    break
            
            if not added:
                break
        
        if len(branches) >= 2:
            return (branches, paths)
        return None
    
    def find_max_clique():
        """Bron-Kerbosch with pivoting to find maximum clique."""
        max_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal max_clique
            if time_up():
                return
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            # Choose pivot
            if P.union(X):
                u = max(P.union(X), key=lambda x: len(adj[x] & (P.union(X))))
            else:
                u = -1
            candidates = P - (adj[u] if u >= 0 else set())
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return max_clique
    
    best_result = None
    best_size = 0
    
    # Strategy 1: Start from max clique and grow
    clique = find_max_clique()
    if clique:
        result = try_greedy_growth(clique)
        if result and len(result[0]) > best_size:
            best_result = result
            best_size = len(result[0])
    
    # Strategy 2: Try multiple random seeds and grow
    random.seed(42)
    num_attempts = 50
    
    for attempt in range(num_attempts):
        if time_up():
            break
        
        # Random seed of size 2-4
        seed_size = random.randint(2, min(4, n-1))
        seed = random.sample(range(n), seed_size)
        
        result = try_greedy_growth(seed)
        if result and len(result[0]) > best_size:
            best_result = result
            best_size = len(result[0])
    
    # Strategy 3: Try systematic pairs as seeds
    if not time_up():
        for i in range(n):
            if time_up():
                break
            for j in range(i+1, n):
                if time_up():
                    break
                result = try_greedy_growth([i, j])
                if result and len(result[0]) > best_size:
                    best_result = result
                    best_size = len(result[0])
    
    # If no result found, fallback to max clique
    if best_result is None:
        if not clique:
            clique = [0, 1]
        branches = list(clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u, v = branches[i], branches[j]
                paths.append([u, v])
        best_result = (branches, paths)
    
    branches, paths_dict = best_result
    
    # Convert paths_dict to list format
    paths_list = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            paths_list.append(paths_dict[(i, j)])
    
    # Ensure types are correct
    branches = [int(b) for b in branches]
    paths_list = [[int(v) for v in p] for p in paths_list]
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
