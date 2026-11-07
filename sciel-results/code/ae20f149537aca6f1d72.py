import time
import itertools
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85.0

    def time_left():
        return deadline - time.time()

    def find_max_clique():
        """Bron-Kerbosch with pivot to find largest maximal clique."""
        max_clique = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique[:] = list(R)
                return
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return max_clique

    def find_all_simple_paths_bfs(src, dst, max_len, avoid_set, adj_list):
        """Find all simple paths from src to dst with length <= max_len, avoiding avoid_set in interior."""
        if src == dst:
            return []
        if dst in avoid_set:
            return []
        paths = []
        # BFS with path tracking
        queue = deque([(src, [src], {src})])
        while queue:
            if time.time() > deadline:
                break
            curr, path, visited = queue.popleft()
            if curr == dst:
                if len(path) - 1 <= max_len:
                    # Check interior avoids avoid_set
                    interior = set(path[1:-1])
                    if not (interior & avoid_set):
                        paths.append(list(path))
                continue
            if len(path) - 1 >= max_len:
                continue
            for nb in sorted(adj_list[curr]):
                if nb == dst:
                    # Allow reaching dst
                    if len(path) + 1 - 1 <= max_len:
                        new_path = path + [nb]
                        interior = set(new_path[1:-1])
                        if not (interior & avoid_set):
                            paths.append(list(new_path))
                    continue
                if nb not in visited and nb not in avoid_set:
                    queue.append((nb, path + [nb], visited | {nb}))
        return paths

    def try_find_disjoint_paths(branches, adj_list, max_path_len=6):
        """
        Try to find internally disjoint paths between all pairs of branches.
        Returns list of paths if successful, else None.
        """
        if len(branches) < 2:
            return []
        
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Generate candidate paths for each pair
        candidates = {}
        for u, v in pairs:
            # Try with increasing max length
            paths = find_all_simple_paths_bfs(u, v, max_path_len, branch_set, adj_list)
            if not paths:
                # Try with longer paths
                paths = find_all_simple_paths_bfs(u, v, max_path_len + 2, branch_set, adj_list)
            # Sort by length, then lexicographically
            paths.sort(key=lambda x: (len(x), x))
            # Limit candidates to keep search tractable
            if len(paths) > 20:
                paths = paths[:20]
            candidates[(u, v)] = paths
        
        # Check if any pair has no candidates
        for u, v in pairs:
            if not candidates[(u, v)]:
                return None
        
        # Order pairs by number of candidates (most constrained first)
        pair_order = sorted(pairs, key=lambda p: len(candidates[p]))
        
        used_internal = set()
        assignment = {}
        
        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            if idx == len(pair_order):
                return True
            
            u, v = pair_order[idx]
            for path in candidates[(u, v)]:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & branch_set:
                    continue
                
                assignment[(u, v)] = path
                used_internal.update(internal)
                
                if backtrack(idx + 1):
                    return True
                
                used_internal.difference_update(internal)
                del assignment[(u, v)]
            
            return False
        
        if backtrack(0):
            # Return paths in original pair order
            return [assignment[p] for p in pairs]
        
        return None

    def is_valid_subdivision(branches, paths, adj_list):
        """Verify that the given branches and paths form a valid subdivision."""
        if len(branches) < 2:
            return False
        branch_set = set(branches)
        # Check all pairs have a path
        pairs = list(itertools.combinations(branches, 2))
        if len(paths) != len(pairs):
            return False
        
        # Check each path
        used_internal = set()
        path_pairs = set()
        for path in paths:
            if not isinstance(path, list) or len(path) < 2:
                return False
            u, v = path[0], path[-1]
            # Check endpoints are branches
            if u not in branch_set or v not in branch_set:
                return False
            if u == v:
                return False
            # Check this pair
            pair = frozenset([u, v])
            if pair in path_pairs:
                return False
            path_pairs.add(pair)
            
            # Check interior vertices
            interior = path[1:-1]
            # Check interior doesn't include branches
            if any(x in branch_set for x in interior):
                return False
            # Check interior is disjoint
            if any(x in used_internal for x in interior):
                return False
            used_internal.update(interior)
            
            # Check consecutive vertices are adjacent
            for i in range(len(path) - 1):
                if path[i + 1] not in adj_list[path[i]]:
                    return False
        
        # Check all pairs are covered
        if path_pairs != set(frozenset(p) for p in pairs):
            return False
        
        return True

    # Phase 1: Find Max Clique
    max_clique = find_max_clique()
    best_size = len(max_clique)
    best_branches = list(max_clique)
    best_paths = None
    
    # If max clique is large enough, it's a valid subdivision
    if best_size >= 2:
        pairs = list(itertools.combinations(best_branches, 2))
        best_paths = [[u, v] for u, v in pairs]
    
    # Phase 2: Try to expand beyond max clique
    # We try to add vertices one by one and check for valid subdivision
    current_branches = list(best_branches)
    
    # Try multiple expansion strategies
    for strategy_round in range(30):
        if time.time() > deadline - 5:
            break
        
        added = False
        # Candidates to add: vertices not in current_branches
        # Sort by degree to prefer well-connected vertices
        candidates_to_add = [v for v in range(n) if v not in set(current_branches)]
        candidates_to_add.sort(key=lambda v: -len(adj[v]))
        
        for v in candidates_to_add:
            if time.time() > deadline - 5:
                break
            
            new_branches = current_branches + [v]
            
            # Determine max_path_len based on size
            # Smaller max_path_len is faster for larger branch sets
            mpl = 4 if len(new_branches) <= 4 else (5 if len(new_branches) <= 6 else 6)
            
            paths = try_find_disjoint_paths(new_branches, adj, max_path_len=mpl)
            
            if paths is not None:
                # Verify
                if is_valid_subdivision(new_branches, paths, adj):
                    current_branches = new_branches
                    best_size = len(current_branches)
                    best_branches = list(current_branches)
                    best_paths = paths
                    added = True
                    break  # Restart candidate selection with new set
        
        if not added:
            # Try a different approach: remove one branch and try to add two others?
            # For now, try swapping: for each current branch, remove it and try to add a non-branch
            swap_improved = False
            for idx, rem_branch in enumerate(list(current_branches)):
                if time.time() > deadline - 5:
                    break
                reduced = [b for i, b in enumerate(current_branches) if i != idx]
                candidates_to_add = [v for v in range(n) if v not in set(reduced) and v != rem_branch]
                candidates_to_add.sort(key=lambda v: -len(adj[v]))
                
                for v in candidates_to_add[:10]:  # Limit to top 10
                    if time.time() > deadline - 5:
                        break
                    new_branches = reduced + [v]
                    mpl = 4 if len(new_branches) <= 4 else (5 if len(new_branches) <= 6 else 6)
                    paths = try_find_disjoint_paths(new_branches, adj, max_path_len=mpl)
                    if paths is not None:
                        if is_valid_subdivision(new_branches, paths, adj):
                            current_branches = new_branches
                            best_size = len(current_branches)
                            best_branches = list(current_branches)
                            best_paths = paths
                            swap_improved = True
                            break
                if swap_improved:
                    break
            
            if not swap_improved:
                # Try removing two branches and adding three?
                # This is getting complex. Try a simpler heuristic: 
                # For each pair of branches to remove, try to add two new ones
                for rem1, rem2 in itertools.combinations(range(len(current_branches)), 2):
                    if time.time() > deadline - 5:
                        break
                    reduced = [b for i, b in enumerate(current_branches) if i not in (rem1, rem2)]
                    candidates_to_add = [v for v in range(n) if v not in set(reduced) and v not in (current_branches[rem1], current_branches[rem2])]
                    candidates_to_add.sort(key=lambda v: -len(adj[v]))
                    
                    for v1, v2 in itertools.combinations(candidates_to_add[:8], 2):
                        if time.time() > deadline - 5:
                            break
                        new_branches = reduced + [v1, v2]
                        mpl = 4 if len(new_branches) <= 4 else (5 if len(new_branches) <= 6 else 6)
                        paths = try_find_disjoint_paths(new_branches, adj, max_path_len=mpl)
                        if paths is not None:
                            if is_valid_subdivision(new_branches, paths, adj):
                                current_branches = new_branches
                                best_size = len(current_branches)
                                best_branches = list(current_branches)
                                best_paths = paths
                                swap_improved = True
                                break
                    if swap_improved:
                        break
                
                if not swap_improved:
                    # No improvement, try starting from scratch with different initial clique
                    # Find all maximal cliques of size >= max_clique_size - 1 and try expanding each
                    all_cliques = []
                    
                    def find_all_max_cliques(min_size):
                        found = []
                        def bronk2(R, P, X):
                            if time.time() > deadline - 10:
                                return
                            if not P and not X:
                                if len(R) >= min_size:
                                    found.append(list(R))
                                return
                            if P or X:
                                pivot = max(P | X, key=lambda v: len(adj[v] & P))
                            else:
                                return
                            candidates = [v for v in P if v not in adj[pivot]]
                            for v in candidates:
                                bronk2(R | {v}, P & adj[v], X & adj[v])
                                P = P - {v}
                                X = X | {v}
                        bronk2(frozenset(), frozenset(range(n)), frozenset())
                        return found
                    
                    # Try expanding from cliques of size max_clique_size - 1
                    if len(max_clique) > 2:
                        smaller_cliques = find_all_max_cliques(len(max_clique) - 1)
                        for clique in smaller_cliques[:10]:  # Limit to 10
                            if time.time() > deadline - 5:
                                break
                            # Try to expand this clique
                            test_branches = list(clique)
                            # Try adding vertices
                            for add_v in [v for v in range(n) if v not in set(test_branches)]:
                                if time.time() > deadline - 5:
                                    break
                                new_branches = test_branches + [add_v]
                                mpl = 4 if len(new_branches) <= 4 else (5 if len(new_branches) <= 6 else 6)
                                paths = try_find_disjoint_paths(new_branches, adj, max_path_len=mpl)
                                if paths is not None:
                                    if is_valid_subdivision(new_branches, paths, adj):
                                        if len(new_branches) > best_size:
                                            current_branches = new_branches
                                            best_size = len(current_branches)
                                            best_branches = list(current_branches)
                                            best_paths = paths
                                        break
                            # Also try adding two
                            for add_v1, add_v2 in itertools.combinations([v for v in range(n) if v not in set(test_branches)], 2):
                                if time.time() > deadline - 5:
                                    break
                                new_branches = test_branches + [add_v1, add_v2]
                                mpl = 4 if len(new_branches) <= 4 else (5 if len(new_branches) <= 6 else 6)
                                paths = try_find_disjoint_paths(new_branches, adj, max_path_len=mpl)
                                if paths is not None:
                                    if is_valid_subdivision(new_branches, paths, adj):
                                        if len(new_branches) > best_size:
                                            current_branches = new_branches
                                            best_size = len(current_branches)
                                            best_branches = list(current_branches)
                                            best_paths = paths
                                        break
                    
                    break  # Break out of main expansion loop
    
    # Final validation and formatting
    if best_size >= 2 and best_paths is not None:
        # Final verification
        if not is_valid_subdivision(best_branches, best_paths, adj):
            # Fallback to max clique if invalid
            if best_size >= 2:
                pairs = list(itertools.combinations(best_branches, 2))
                best_paths = [[u, v] for u, v in pairs]
            else:
                if P['edges']:
                    u, v = P['edges'][0]
                    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
                else:
                    return {'branches': [0, 1], 'paths': [[0, 1]]}
        
        branches = [int(b) for b in best_branches]
        formatted_paths = []
        for p in best_paths:
            formatted_paths.append([int(x) for x in p])
        
        return {'branches': branches, 'paths': formatted_paths}
    
    # Fallback: Any edge
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
