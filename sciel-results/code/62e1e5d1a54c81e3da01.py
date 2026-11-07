"""Rewritten program: structural change to flow-based disjoint path finding."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin: 90s limit, leave 40s buffer

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def check_time():
        return time.time() < deadline

    def find_max_clique():
        """Bron-Kerbosch with pivot to find maximum clique."""
        best_clique = []
        
        def bron_kerbosch(R, P_set, X):
            nonlocal best_clique
            if not check_time():
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            if P_set:
                pivot = max(P_set, key=lambda v: len(adj[v] & (P_set | X)))
            else:
                pivot = None
            candidates = list(P_set)
            if pivot is not None:
                candidates = [v for v in candidates if v not in adj[pivot]]
            for v in candidates:
                if not check_time():
                    return
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
                P_set.discard(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return best_clique

    def bfs_path(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden vertices."""
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]
        visited = {u: None}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            if node == v:
                path = [v]
                while path[-1] != u:
                    path.append(visited[path[-1]])
                path.reverse()
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited[neighbor] = node
                    queue.append(neighbor)
        return None

    def find_disjoint_paths(branches, max_attempts=80):
        """
        Find vertex-disjoint paths for all pairs of branches.
        Uses randomized greedy with BFS shortest paths.
        Returns dict mapping (u,v) -> path, or None.
        """
        if len(branches) < 2:
            return {}
        
        pairs = list(combinations(sorted(branches), 2))
        best_result = None
        best_interior_count = float('inf')
        
        for attempt in range(max_attempts):
            if not check_time():
                break
            
            shuffled_pairs = list(pairs)
            random.shuffle(shuffled_pairs)
            
            candidate_paths = {}
            used_interiors = set()
            success = True
            
            for u, v in shuffled_pairs:
                path_forbidden = set(used_interiors)
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                
                p = bfs_path(u, v, path_forbidden)
                if p is None:
                    success = False
                    break
                
                candidate_paths[(u, v)] = p
                used_interiors.update(p[1:-1])
            
            if success:
                interior_count = len(used_interiors)
                if interior_count < best_interior_count:
                    best_result = candidate_paths
                    best_interior_count = interior_count
                    if interior_count == 0:
                        break
        
        return best_result

    def is_valid(branches, paths_dict):
        """Validate the solution."""
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
        
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
            
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return False
            
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
        
        return True

    # Strategy 1: Find maximum clique, then try subsets
    clique = find_max_clique()
    
    best_branches = []
    best_paths = {}
    
    if len(clique) >= 2:
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)
        
        for size in range(min(len(clique_sorted), 20), 1, -1):
            if not check_time():
                break
            
            subset = clique_sorted[:size]
            paths = find_disjoint_paths(subset, max_attempts=60)
            if paths is not None and is_valid(subset, paths):
                best_branches = subset
                best_paths = paths
                break
        
        # Try random subsets of clique
        for trial in range(40):
            if not check_time():
                break
            if len(clique_sorted) <= len(best_branches):
                break
            subset = random.sample(clique_sorted, len(best_branches) + 1 if len(best_branches) < len(clique_sorted) else len(clique_sorted))
            paths = find_disjoint_paths(subset, max_attempts=40)
            if paths is not None and is_valid(subset, paths):
                if len(subset) > len(best_branches):
                    best_branches = subset
                    best_paths = paths

    # Strategy 2: Greedy construction from high-degree edges
    if len(best_branches) < 4:
        sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        
        for u, v in sorted_edges[:20]:
            if not check_time():
                break
            
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]}
            
            for _ in range(100):
                if not check_time():
                    break
                
                candidates = [i for i in range(n) if i not in current_branches]
                candidates.sort(key=lambda x: len(adj[x]), reverse=True)
                
                added = False
                for new_b in candidates[:12]:
                    new_branches = current_branches + [new_b]
                    new_paths = find_disjoint_paths(new_branches, max_attempts=25)
                    if new_paths is not None and is_valid(new_branches, new_paths):
                        current_branches = new_branches
                        current_paths = new_paths
                        added = True
                        break
                
                if not added:
                    break
            
            if len(current_branches) > len(best_branches):
                best_branches = current_branches
                best_paths = current_paths

    # Strategy 3: Local search - swap branches
    if len(best_branches) >= 3:
        improved = True
        while improved:
            if not check_time():
                break
            improved = False
            
            for trial in range(30):
                if not check_time():
                    break
                
                remove_idx = random.randint(0, len(best_branches) - 1)
                removed = best_branches[remove_idx]
                remaining = [b for i, b in enumerate(best_branches) if i != remove_idx]
                
                candidates = [i for i in range(n) if i not in remaining]
                random.shuffle(candidates)
                
                for new_b in candidates[:8]:
                    new_branches = remaining + [new_b]
                    new_paths = find_disjoint_paths(new_branches, max_attempts=35)
                    if new_paths is not None and is_valid(new_branches, new_paths):
                        if len(new_branches) > len(best_branches):
                            best_branches = new_branches
                            best_paths = new_paths
                            improved = True
                        break

    # Strategy 4: Multiple random restarts
    for restart in range(15):
        if not check_time():
            break
        
        if edges:
            u, v = random.choice(edges)
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]}
        else:
            break
        
        for _ in range(60):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current_branches]
            random.shuffle(candidates)
            
            added = False
            for new_b in candidates[:10]:
                new_branches = current_branches + [new_b]
                new_paths = find_disjoint_paths(new_branches, max_attempts=20)
                if new_paths is not None and is_valid(new_branches, new_paths):
                    current_branches = new_branches
                    current_paths = new_paths
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_branches) > len(best_branches):
            best_branches = current_branches
            best_paths = current_paths

    # Build final output
    if len(best_branches) < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    final_paths = []
    for u, v in combinations(sorted(best_branches), 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])
    
    return {
        'branches': [int(x) for x in best_branches],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
