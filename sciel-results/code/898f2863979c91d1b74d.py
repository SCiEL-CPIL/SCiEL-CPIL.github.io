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
    
    def bfs_shortest_path(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden nodes."""
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]
        
        visited = {u: None}
        queue = deque([u])
        
        while queue:
            node = queue.popleft()
            if node == v:
                # Reconstruct path
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
    
    def find_disjoint_paths(branches, adj, forbidden_base=None):
        """
        Given a set of branch vertices, find vertex-disjoint paths between all pairs.
        Returns dict mapping (u,v) -> path, or None if not possible.
        Uses a greedy approach with randomization.
        """
        if len(branches) < 2:
            return {}
        
        pairs = list(combinations(branches, 2))
        paths = {}
        
        # Try multiple random orderings of pairs
        for attempt in range(50):
            if not check_time():
                return None
            
            random.shuffle(pairs)
            candidate_paths = {}
            used_interiors = set()
            
            if forbidden_base:
                used_interiors = set(forbidden_base)
            
            success = True
            for u, v in pairs:
                # Build forbidden set for this path
                path_forbidden = set(used_interiors)
                # All branches except u and v are forbidden
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                
                # Find path
                p = bfs_shortest_path(u, v, path_forbidden)
                if p is None:
                    success = False
                    break
                
                candidate_paths[(u, v)] = p
                used_interiors.update(p[1:-1])
            
            if success:
                return candidate_paths
        
        return None
    
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
            if set(path) != set([path[0], path[-1]]) and len(set(path)) != len(path):
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
    
    def try_add_branch(branches, paths_dict, new_branch):
        """Try to add a new branch to the existing configuration."""
        new_branches = branches + [new_branch]
        
        # Find paths from new_branch to all existing branches
        # while keeping existing paths intact
        new_paths = dict(paths_dict)
        
        # Collect existing interior vertices
        existing_interiors = set()
        for key, path in paths_dict.items():
            existing_interiors.update(path[1:-1])
        
        # Find new paths
        new_pairs = [(new_branch, b) for b in branches]
        random.shuffle(new_pairs)
        
        current_forbidden = set(existing_interiors)
        found_new_paths = {}
        
        for u, v in new_pairs:
            path_forbidden = set(current_forbidden)
            for b in new_branches:
                if b != u and b != v:
                    path_forbidden.add(b)
            
            p = bfs_shortest_path(u, v, path_forbidden)
            if p is None:
                return None
            found_new_paths[(u, v)] = p
            current_forbidden.update(p[1:-1])
        
        for k, p in found_new_paths.items():
            new_paths[k] = p
        
        if is_valid_solution(new_branches, new_paths):
            return new_branches, new_paths
        return None
    
    def local_search_improve(branches, paths_dict):
        """Try to improve the solution by swapping branches."""
        best = (branches, paths_dict)
        best_size = len(branches)
        
        for _ in range(200):
            if not check_time():
                break
            
            # Try removing a random branch and adding a different one
            if len(branches) < 3:
                break
            
            remove_idx = random.randint(0, len(branches) - 1)
            removed = branches[remove_idx]
            remaining = [b for i, b in enumerate(branches) if i != remove_idx]
            
            # Remove paths involving the removed branch
            reduced_paths = {}
            for key, path in paths_dict.items():
                if removed not in key:
                    reduced_paths[key] = path
            
            # Try to add a new branch from remaining nodes
            candidates = [i for i in range(n) if i not in remaining]
            random.shuffle(candidates)
            
            for new_b in candidates:
                result = try_add_branch(remaining, reduced_paths, new_b)
                if result is not None:
                    new_branches, new_paths = result
                    if len(new_branches) > best_size:
                        best = (new_branches, new_paths)
                        best_size = len(new_branches)
                        branches, paths_dict = new_branches, new_paths
                    break
        
        return best
    
    # Strategy 1: Find maximum clique, then try to find disjoint paths
    # This is the key insight: if we have a clique, we can potentially use
    # the clique structure to find disjoint paths
    
    # Find maximum clique using branch and bound
    def max_clique():
        """Find maximum clique in the graph."""
        # Use Bron-Kerbosch with pivot
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
    
    # Find maximum clique
    clique = max_clique()
    
    # Try to find disjoint paths for the clique
    best_branches = []
    best_paths = {}
    
    if len(clique) >= 2:
        # Sort clique by degree (descending) for better path finding
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)
        
        # Try different subsets of the clique (starting with full clique)
        for size in range(min(len(clique_sorted), 12), 1, -1):
            if not check_time():
                break
            
            # Try all subsets of this size from the top of the sorted clique
            # For efficiency, just try the first few
            subset = clique_sorted[:size]
            paths = find_disjoint_paths(subset, adj)
            if paths is not None:
                if is_valid_solution(subset, paths):
                    best_branches = subset
                    best_paths = paths
                    break
    
    # If clique approach didn't work well, try greedy construction
    if len(best_branches) < 2:
        # Greedy: start with a high-degree edge, then add vertices
        # Sort edges by sum of degrees
        sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        
        for u, v in sorted_edges[:50]:
            if not check_time():
                break
            
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]}
            
            # Try to add more branches
            for _ in range(200):
                if not check_time():
                    break
                
                candidates = [i for i in range(n) if i not in current_branches]
                # Sort by degree
                candidates.sort(key=lambda x: len(adj[x]), reverse=True)
                
                added = False
                for new_b in candidates[:20]:
                    result = try_add_branch(current_branches, current_paths, new_b)
                    if result is not None:
                        current_branches, current_paths = result
                        added = True
                        break
                
                if not added:
                    break
            
            if len(current_branches) > len(best_branches):
                best_branches = current_branches
                best_paths = current_paths
    
    # Local search to improve
    if len(best_branches) >= 2:
        best_branches, best_paths = local_search_improve(best_branches, best_paths)
    
    # Multiple random restarts for better exploration
    for restart in range(30):
        if not check_time():
            break
        
        # Random initialization
        if edges:
            u, v = random.choice(edges)
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]}
        else:
            break
        
        # Greedy expansion
        for _ in range(100):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current_branches]
            random.shuffle(candidates)
            
            added = False
            for new_b in candidates[:15]:
                result = try_add_branch(current_branches, current_paths, new_b)
                if result is not None:
                    current_branches, current_paths = result
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_branches) > len(best_branches):
            best_branches = current_branches
            best_paths = current_paths
    
    # Final local search
    if len(best_branches) >= 2:
        best_branches, best_paths = local_search_improve(best_branches, best_paths)
    
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
    for u, v in combinations(best_branches, 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])
    
    return {
        'branches': [int(x) for x in best_branches],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
