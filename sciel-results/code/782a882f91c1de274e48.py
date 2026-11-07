"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s limit
    
    n = P['n']
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def is_adjacent(u, v):
        return v in adj[u]
    
    def find_path_bfs(src, dst, forbidden):
        """Find a simple path from src to dst avoiding forbidden vertices (except src, dst)."""
        if src == dst:
            return None
        if dst in forbidden:
            return None
        if src in forbidden and src != dst:
            return None
        
        # BFS to find shortest path
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        
        while queue:
            curr = queue.popleft()
            if curr == dst:
                # Reconstruct path
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for neighbor in adj[curr]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        
        return None
    
    def try_subdivision(branches, deadline):
        """
        Given a set of branch vertices, try to find vertex-disjoint paths between all pairs.
        Returns (paths_dict, success) where paths_dict maps (u,v) -> path.
        """
        branches = sorted(branches)
        pairs = list(combinations(branches, 2))
        
        # Order pairs by "difficulty" - use heuristic: pairs with fewer common neighbors first
        def pair_difficulty(pair):
            u, v = pair
            # Count neighbors of u not in branches
            nu = len([x for x in adj[u] if x not in branches])
            nv = len([x for x in adj[v] if x not in branches])
            # Pairs where endpoints have fewer alternatives are more constrained
            return nu + nv
        
        pairs_sorted = sorted(pairs, key=pair_difficulty)
        
        assigned_paths = {}
        used_interior = set()
        
        for u, v in pairs_sorted:
            if time.time() > deadline:
                return None, False
            
            # Find path avoiding all branch vertices except u and v, and all used interior vertices
            forbidden = set(branches) - {u, v}
            forbidden.update(used_interior)
            
            path = find_path_bfs(u, v, forbidden)
            if path is None:
                return None, False
            
            # Check that interior vertices don't conflict
            interior = set(path[1:-1])
            if interior & used_interior:
                return None, False
            if interior & (set(branches) - {u, v}):
                return None, False
            
            assigned_paths[(u, v)] = path
            used_interior.update(interior)
        
        # Verify all paths are valid
        for (u, v), path in assigned_paths.items():
            if path[0] != u or path[-1] != v:
                return None, False
            for i in range(len(path) - 1):
                if not is_adjacent(path[i], path[i+1]):
                    return None, False
            interior = set(path[1:-1])
            if interior & set(branches):
                return None, False
        
        return assigned_paths, True
    
    def try_subdivision_with_backtrack(branches, deadline):
        """
        More sophisticated approach: try to assign paths with some backtracking on path choice.
        Uses BFS but tries multiple paths if the first one leads to dead end.
        """
        branches = sorted(branches)
        pairs = list(combinations(branches, 2))
        
        # Sort by difficulty
        def pair_difficulty(pair):
            u, v = pair
            nu = len([x for x in adj[u] if x not in branches])
            nv = len([x for x in adj[v] if x not in branches])
            return nu + nv
        
        pairs_sorted = sorted(pairs, key=pair_difficulty)
        
        # Recursive backtracking with limited depth
        def backtrack(idx, used_interior):
            if time.time() > deadline:
                return None
            if idx == len(pairs_sorted):
                return {}
            
            u, v = pairs_sorted[idx]
            forbidden = set(branches) - {u, v}
            forbidden.update(used_interior)
            
            # Find multiple candidate paths using BFS with different orderings
            # For simplicity, use BFS to get one path, and if it fails later, we'd need to backtrack
            # For now, just use the BFS path
            path = find_path_bfs(u, v, forbidden)
            if path is None:
                return None
            
            interior = set(path[1:-1])
            new_used = used_interior | interior
            
            result = backtrack(idx + 1, new_used)
            if result is not None:
                result[(u, v)] = path
                return result
            
            return None
        
        return backtrack(0, set())
    
    # First, find a good set of branches using a greedy/randomized approach
    # Key insight: branches don't need to form a clique; they just need disjoint paths between all pairs
    
    best_result = None
    best_count = 2
    
    # Start with a simple valid solution
    u, v = P['edges'][0]
    best_result = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    best_count = 2
    
    # Try to build larger subdivisions
    # Strategy: start with small sets and grow
    
    def try_add_vertex(current_branches, candidate, deadline):
        """Try to add a new branch vertex to an existing subdivision."""
        new_branches = sorted(current_branches + [candidate])
        paths, ok = try_subdivision(new_branches, deadline)
        if ok:
            return new_branches, paths
        return None, None
    
    # Randomized search
    random.seed(42)
    
    # Try all pairs first to establish a baseline
    for i, j in combinations(range(n), 2):
        if time.time() > deadline:
            break
        if is_adjacent(i, j):
            # Direct edge is always a valid path
            if 2 > best_count:
                best_count = 2
                best_result = {'branches': [int(i), int(j)], 'paths': [[int(i), int(j)]]}
    
    # Now try to build larger sets
    # Start from pairs and try to add vertices
    active_sets = []
    
    # Initialize with some promising pairs
    for i in range(n):
        for j in range(i+1, n):
            if is_adjacent(i, j):
                active_sets.append((i, j))
    
    # Greedy growth: try to add vertices to existing sets
    for _ in range(100):  # Limited iterations
        if time.time() > deadline:
            break
        
        # Pick a random active set
        if not active_sets:
            break
        
        idx = random.randint(0, len(active_sets) - 1)
        current = list(active_sets[idx])
        
        # Try to add a random vertex
        candidates = [v for v in range(n) if v not in current]
        random.shuffle(candidates)
        
        added = False
        for cand in candidates[:5]:  # Try up to 5 candidates
            if time.time() > deadline:
                break
            new_branches, paths = try_add_vertex(current, cand, deadline)
            if new_branches is not None:
                if len(new_branches) > best_count:
                    best_count = len(new_branches)
                    paths_list = []
                    for (u, v) in combinations(sorted(new_branches), 2):
                        paths_list.append([int(x) for x in paths[(u, v)]])
                    best_result = {'branches': [int(x) for x in new_branches], 'paths': paths_list}
                
                # Replace the old set with the new one
                active_sets[idx] = new_branches
                added = True
                break
        
        if not added:
            # Remove this set if we can't grow it
            active_sets.pop(idx)
    
    # Also try some random sets of size 3-6
    for size in range(3, min(7, n+1)):
        if time.time() > deadline:
            break
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            rand_set = random.sample(range(n), size)
            paths, ok = try_subdivision(rand_set, deadline)
            if ok and size > best_count:
                best_count = size
                paths_list = []
                for (u, v) in combinations(sorted(rand_set), 2):
                    paths_list.append([int(x) for x in paths[(u, v)]])
                best_result = {'branches': [int(x) for x in rand_set], 'paths': paths_list}
    
    # Try to improve by local search on the best set found
    if best_count >= 3:
        current_branches = best_result['branches']
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Try swapping one vertex
            for i in range(len(current_branches)):
                if time.time() > deadline:
                    break
                for cand in range(n):
                    if cand in current_branches:
                        continue
                    new_branches = current_branches[:i] + [cand] + current_branches[i+1:]
                    new_branches = sorted(new_branches)
                    paths, ok = try_subdivision(new_branches, deadline)
                    if ok:
                        # Same size, but might allow further growth
                        # Try to grow from here
                        grown = False
                        for add in range(n):
                            if add in new_branches:
                                continue
                            nb, np = try_add_vertex(new_branches, add, deadline)
                            if nb is not None:
                                if len(nb) > best_count:
                                    best_count = len(nb)
                                    paths_list = []
                                    for (u, v) in combinations(sorted(nb), 2):
                                        paths_list.append([int(x) for x in np[(u, v)]])
                                    best_result = {'branches': [int(x) for x in nb], 'paths': paths_list}
                                current_branches = nb
                                grown = True
                                break
                        if grown:
                            break
    
    # Final verification of best result
    if best_result is not None:
        branches = best_result['branches']
        paths = best_result['paths']
        
        # Verify
        branch_set = set(branches)
        used_interior = set()
        valid = True
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            if path[0] not in branch_set or path[-1] not in branch_set:
                valid = False
                break
            if path[0] == path[-1]:
                valid = False
                break
            for i in range(len(path)-1):
                if not is_adjacent(path[i], path[i+1]):
                    valid = False
                    valid = False
                    break
            if not valid:
                break
            interior = set(path[1:-1])
            if interior & branch_set:
                valid = False
                break
            if interior & used_interior:
                valid = False
                break
            used_interior.update(interior)
        
        if valid:
            return {'branches': [int(b) for b in branches], 'paths': [[int(v) for v in p] for p in paths]}
    
    # Fallback
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
