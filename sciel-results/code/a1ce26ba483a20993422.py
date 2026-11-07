"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def check_time():
        return time.time() < deadline

    def bfs_shortest_paths(u, v, forbidden, limit=10):
        """
        Find up to 'limit' shortest simple paths from u to v avoiding forbidden nodes.
        Returns a list of paths.
        """
        if u in forbidden or v in forbidden:
            return []
        if u == v:
            return [[u, v]]
        
        # Standard BFS to find shortest path length
        visited = {u: None}
        queue = deque([u])
        dist = {u: 0}
        
        while queue:
            node = queue.popleft()
            if node == v:
                break
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited[neighbor] = node
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        
        if v not in visited:
            return []
        
        shortest_len = dist[v]
        
        # DFS to find all paths of length shortest_len
        paths = []
        def dfs(node, current_path):
            if len(paths) >= limit:
                return
            if node == v:
                if len(current_path) - 1 == shortest_len:
                    paths.append(list(current_path))
                return
            for neighbor in adj[node]:
                if neighbor not in current_path and neighbor not in forbidden:
                    if dist.get(neighbor, float('inf')) == dist[node] + 1:
                        current_path.append(neighbor)
                        dfs(neighbor, current_path)
                        current_path.pop()
            
        dfs(u, [u])
        return paths

    def is_valid_solution(branches, paths_dict):
        """Validate that the solution satisfies all constraints."""
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
        
        # Check all pairs have paths
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict and (v, u) not in paths_dict:
                return False
        
        used_interiors = set()
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] not in (u, v) or path[-1] not in (u, v):
                return False
            if path[0] == path[-1]:
                return False
            if len(set(path)) != len(path):
                return False
            
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
            # Check interiors
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
        
        return True

    def find_disjoint_paths_backtracking(branches, max_nodes_in_path=10):
        """
        Find vertex-disjoint paths for all pairs in branches using backtracking.
        Returns dict mapping (u,v) -> path, or None.
        """
        if len(branches) < 2:
            return {}
        
        pairs = list(combinations(sorted(branches), 2))
        
        # Sort pairs by "tightness" - pairs with fewer available shortest paths first
        pair_options = {}
        for u, v in pairs:
            # Forbidden: other branches
            forbidden = set(branches) - {u, v}
            opts = bfs_shortest_paths(u, v, forbidden, limit=5)
            if not opts:
                return None
            pair_options[(u, v)] = opts
        
        # Order pairs: those with fewer options first
        ordered_pairs = sorted(pairs, key=lambda p: len(pair_options[p]))
        
        paths_dict = {}
        used_interiors = set()
        
        def backtrack(idx):
            if idx == len(ordered_pairs):
                return True
            
            u, v = ordered_pairs[idx]
            forbidden = set(used_interiors) | (set(branches) - {u, v})
            
            # Filter options that are still valid
            valid_options = []
            for p in pair_options[(u, v)]:
                # Check if path intersects used interiors
                if not any(node in used_interiors for node in p[1:-1]):
                    valid_options.append(p)
            
            # Shuffle for randomness
            random.shuffle(valid_options)
            
            for p in valid_options:
                paths_dict[(u, v)] = p
                new_used = set(used_interiors)
                new_used.update(p[1:-1])
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                del paths_dict[(u, v)]
            
            return False
        
        if backtrack(0):
            return paths_dict
        return None

    def find_disjoint_paths_greedy(branches, num_attempts=20):
        """
        Greedy randomized approach.
        """
        if len(branches) < 2:
            return {}
        
        pairs = list(combinations(branches, 2))
        
        for attempt in range(num_attempts):
            if not check_time():
                return None
            
            random.shuffle(pairs)
            candidate_paths = {}
            used_interiors = set()
            
            success = True
            for u, v in pairs:
                path_forbidden = set(used_interiors)
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                
                # Get a random shortest path
                opts = bfs_shortest_paths(u, v, path_forbidden, limit=1)
                if not opts:
                    success = False
                    break
                
                p = opts[0]
                candidate_paths[(u, v)] = p
                used_interiors.update(p[1:-1])
            
            if success:
                return candidate_paths
        
        return None

    def try_find_paths_for_branches(branches):
        """
        Try to find paths for a set of branches.
        First try backtracking (exact for small k), then greedy randomized.
        """
        # Backtracking is better for small sets, greedy for larger or as fallback
        if len(branches) <= 5:
            res = find_disjoint_paths_backtracking(branches)
            if res is not None:
                return res
        
        # Fallback to greedy randomized
        res = find_disjoint_paths_greedy(branches, num_attempts=50)
        return res

    def max_clique():
        """Find maximum clique in the graph using Bron-Kerbosch with pivot."""
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if not check_time():
                return
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & (P | X)))
            else:
                pivot = None
            
            candidates = set(P)
            if pivot is not None:
                candidates -= adj[pivot]
            
            for v in list(candidates):
                if not check_time():
                    return
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return best_clique

    def greedy_expand(start_branches):
        """
        Start with initial branches and greedily add more.
        Returns (branches, paths_dict).
        """
        current_branches = list(start_branches)
        
        current_paths = try_find_paths_for_branches(current_branches)
        if current_paths is None:
            return [], {}
        
        # Greedily add more branches
        for _ in range(300):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current_branches]
            if not candidates:
                break
            
            # Sort candidates by degree (descending)
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            added = False
            for new_b in candidates[:20]:
                # Try to add new branch
                new_branches = current_branches + [new_b]
                
                # Find paths for all pairs
                new_paths = try_find_paths_for_branches(new_branches)
                
                if new_paths is not None:
                    if is_valid_solution(new_branches, new_paths):
                        current_branches = new_branches
                        current_paths = new_paths
                        added = True
                        break
            
            if not added:
                break
        
        return current_branches, current_paths

    def local_search(branches, paths_dict):
        """
        Try to improve the solution by swapping branches.
        Returns improved (branches, paths_dict).
        """
        best = (list(branches), dict(paths_dict))
        best_size = len(branches)
        
        for iteration in range(200):
            if not check_time():
                break
            
            if len(branches) < 3:
                break
            
            # Try removing a random branch and adding a different one
            remove_idx = random.randint(0, len(branches) - 1)
            removed = branches[remove_idx]
            remaining = [b for i, b in enumerate(branches) if i != remove_idx]
            
            # Try to add a new branch
            candidates = [i for i in range(n) if i not in remaining]
            random.shuffle(candidates)
            
            improved = False
            for new_b in candidates[:20]:
                new_branches = remaining + [new_b]
                new_paths = try_find_paths_for_branches(new_branches)
                
                if new_paths is not None and is_valid_solution(new_branches, new_paths):
                    if len(new_branches) > best_size:
                        best = (new_branches, new_paths)
                        best_size = len(new_branches)
                        branches, paths_dict = new_branches, new_paths
                    improved = True
                    break
            
            if not improved:
                continue
        
        return best

    # Strategy 1: Use maximum clique
    clique = max_clique()
    
    best_branches = []
    best_paths = {}
    
    if len(clique) >= 2:
        # Sort clique by degree (descending)
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)
        
        # Try subsets from largest to smallest
        for size in range(min(len(clique_sorted), 15), 1, -1):
            if not check_time():
                break
            
            subset = clique_sorted[:size]
            paths = try_find_paths_for_branches(subset)
            if paths is not None and is_valid_solution(subset, paths):
                if len(subset) > len(best_branches):
                    best_branches = subset
                    best_paths = paths
                break
    
    # Strategy 2: Greedy expansion from various starting points
    # Start with high-degree edges
    sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
    
    for u, v in sorted_edges[:30]:
        if not check_time():
            break
        
        branches, paths = greedy_expand([u, v])
        if len(branches) > len(best_branches):
            best_branches = branches
            best_paths = paths
    
    # Strategy 3: Random restarts
    for restart in range(100):
        if not check_time():
            break
        
        # Random initialization with an edge
        if edges:
            u, v = random.choice(edges)
            branches, paths = greedy_expand([u, v])
            if len(branches) > len(best_branches):
                best_branches = branches
                best_paths = paths
    
    # Strategy 4: Local search improvement
    if len(best_branches) >= 2:
        best_branches, best_paths = local_search(best_branches, best_paths)
    
    # Strategy 5: More aggressive search - try to add branches to current best
    if len(best_branches) >= 2 and check_time():
        current = list(best_branches)
        current_paths = dict(best_paths)
        
        for _ in range(100):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current]
            if not candidates:
                break
            
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            added = False
            for new_b in candidates[:10]:
                new_branches = current + [new_b]
                new_paths = try_find_paths_for_branches(new_branches)
                
                if new_paths is not None and is_valid_solution(new_branches, new_paths):
                    current = new_branches
                    current_paths = new_paths
                    added = True
                    break
            
            if not added:
                break
        
        if len(current) > len(best_branches):
            best_branches = current
            best_paths = current_paths
    
    # Build final output
    if len(best_branches) < 2:
        # Fallback
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    # Construct paths list in consistent order
    final_paths = []
    for u, v in combinations(sorted(best_branches), 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])
    
    return {
        'branches': [int(x) for x in sorted(best_branches)],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
