import time
import sys
from itertools import combinations
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_clique(adj):
    n = P['n']
    best_clique = []
    
    def bronk(R, P_set, X_set):
        nonlocal best_clique
        if not P_set and not X_set:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        U = P_set | X_set
        if U:
            pivot = max(U, key=lambda v: len(adj[v] & P_set))
            candidates = [v for v in P_set if v not in adj[pivot]]
        else:
            candidates = list(P_set)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X_set & adj[v]
            bronk(new_R, new_P, new_X)
            P_set.discard(v)
            X_set.add(v)
            
    # Use degeneracy ordering for better performance
    deg_order = sorted(range(n), key=lambda v: len(adj[v]))
    for i, v in enumerate(deg_order):
        P_set = set(adj[v]) & set(deg_order[i+1:])
        X_set = set()
        bronk({v}, P_set, X_set)
    
    if not best_clique:
        bronk(set(), set(range(n)), set())
    return best_clique

def _find_path_length2(adj, u, v, forbidden):
    """Find a path of length exactly 2 from u to v avoiding forbidden vertices."""
    # u - w - v, where w is not in forbidden and w != u, w != v
    for w in adj[u]:
        if w == v:
            continue
        if w in forbidden:
            continue
        if v in adj[w]:
            return [u, w, v]
    return None

def _find_path_length3(adj, u, v, forbidden):
    """Find a path of length exactly 3 from u to v avoiding forbidden vertices."""
    # u - a - b - v
    for a in adj[u]:
        if a == v or a in forbidden:
            continue
        for b in adj[a]:
            if b == u or b == v or b in forbidden:
                continue
            if v in adj[b]:
                return [u, a, b, v]
    return None

def _find_path_bfs(adj, u, v, forbidden, max_len=4):
    """Find a simple path from u to v with length <= max_len using BFS, avoiding forbidden vertices."""
    if u == v:
        return None
    if u in forbidden or v in forbidden:
        return None
    
    # BFS with path reconstruction
    from collections import deque
    queue = deque()
    queue.append((u, [u]))
    visited = {u}
    
    while queue:
        node, path = queue.popleft()
        if len(path) > max_len:
            continue
        if node == v:
            return path
        for neighbor in adj[node]:
            if neighbor in visited or neighbor in forbidden:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None

def _find_path(adj, u, v, forbidden, max_len=4):
    """Find a simple path from u to v with length <= max_len, avoiding forbidden vertices."""
    # First try direct edge
    if v in adj[u] and u not in forbidden and v not in forbidden:
        return [u, v]
    
    # Try BFS for paths up to max_len
    path = _find_path_bfs(adj, u, v, forbidden, max_len)
    if path:
        return path
    
    return None

def _check_subdivision_fast(adj, branches, time_limit=1.0):
    """
    Fast check if a clique subdivision exists for the given set of branches.
    Uses greedy assignment with limited backtracking.
    """
    start_time = time.time()
    branch_set = set(branches)
    pairs = list(combinations(branches, 2))
    
    # Sort pairs by number of available paths (ascending) for constrained-first
    pair_info = []
    for u, v in pairs:
        forbidden = branch_set - {u, v}
        # Count potential paths
        count = 0
        if v in adj[u]:
            count += 1
        for w in adj[u]:
            if w != v and w not in forbidden and v in adj[w]:
                count += 1
        # Also count longer paths via BFS
        if count == 0:
            path = _find_path_bfs(adj, u, v, forbidden, 4)
            if path:
                count = 1
        pair_info.append((count, (u, v)))
    
    if any(c == 0 for c, _ in pair_info):
        return False
    
    pair_info.sort(key=lambda x: x[0])
    sorted_pairs = [p for _, p in pair_info]
    
    # Greedy with limited backtracking
    def solve(idx, used_interiors):
        if time.time() - start_time > time_limit:
            return False
        
        if idx == len(sorted_pairs):
            return True
            
        u, v = sorted_pairs[idx]
        forbidden = branch_set - {u, v}
        forbidden.update(used_interiors)
        
        # Try paths of increasing length
        for max_len in [1, 2, 3, 4]:
            path = _find_path(adj, u, v, forbidden, max_len)
            if path:
                new_used = set(used_interiors)
                new_used.update(path[1:-1])
                if solve(idx + 1, new_used):
                    return True
        
        return False

    return solve(0, set())

def construct():
    start_time = time.time()
    deadline = start_time + 40.0  # More time for better search
    
    adj = _build_adj()
    
    # 1. Find Max Clique
    clique = _find_max_clique(adj)
    current_branches = set(clique)
    
    # 2. Try to expand greedily with multiple attempts
    # Try adding vertices one by one, multiple passes
    for _pass in range(3):
        if time.time() > deadline - 10.0:
            break
        added = False
        for v in range(P['n']):
            if v in current_branches:
                continue
            if time.time() > deadline - 10.0:
                break
            new_branches = current_branches | {v}
            if _check_subdivision_fast(adj, list(new_branches), time_limit=1.0):
                current_branches.add(v)
                added = True
        if not added:
            break
    
    # 3. Construct paths for the final set of branches
    final_branches = list(current_branches)
    branch_set = set(final_branches)
    pairs = list(combinations(final_branches, 2))
    
    # Sort pairs for efficiency
    pair_info = []
    for u, v in pairs:
        forbidden = branch_set - {u, v}
        count = 0
        if v in adj[u]:
            count += 1
        for w in adj[u]:
            if w != v and w not in forbidden and v in adj[w]:
                count += 1
        if count == 0:
            path = _find_path_bfs(adj, u, v, forbidden, 4)
            if path:
                count = 1
        pair_info.append((count, (u, v)))
    
    pair_info.sort(key=lambda x: x[0])
    sorted_pairs = [p for _, p in pair_info]
    
    # Find paths with backtracking
    def find_paths(idx, used_interiors, current_paths):
        if time.time() > deadline - 3.0:
            return False
        if idx == len(sorted_pairs):
            return True
            
        u, v = sorted_pairs[idx]
        forbidden = branch_set - {u, v}
        forbidden.update(used_interiors)
        
        # Try paths of increasing length
        for max_len in [1, 2, 3, 4]:
            path = _find_path(adj, u, v, forbidden, max_len)
            if path:
                new_used = set(used_interiors)
                new_used.update(path[1:-1])
                current_paths.append(path)
                if find_paths(idx + 1, new_used, current_paths):
                    return True
                current_paths.pop()
        
        return False
        
    found_paths = []
    if not find_paths(0, set(), found_paths):
        # Fallback: try with smaller branch set
        # Remove branches until we find a valid subdivision
        for remove_count in range(1, len(final_branches)):
            if time.time() > deadline - 3.0:
                break
            for combo in combinations(final_branches, remove_count):
                reduced = set(final_branches) - set(combo)
                if len(reduced) < 2:
                    continue
                test_branches = list(reduced)
                test_set = set(test_branches)
                test_pairs = list(combinations(test_branches, 2))
                
                test_pair_info = []
                for u, v in test_pairs:
                    forbidden = test_set - {u, v}
                    count = 0
                    if v in adj[u]:
                        count += 1
                    for w in adj[u]:
                        if w != v and w not in forbidden and v in adj[w]:
                            count += 1
                    if count == 0:
                        path = _find_path_bfs(adj, u, v, forbidden, 4)
                        if path:
                            count = 1
                    test_pair_info.append((count, (u, v)))
                
                if any(c == 0 for c, _ in test_pair_info):
                    continue
                
                test_pair_info.sort(key=lambda x: x[0])
                test_sorted_pairs = [p for _, p in test_pair_info]
                
                def find_paths_test(idx, used_interiors, current_paths):
                    if time.time() > deadline - 3.0:
                        return False
                    if idx == len(test_sorted_pairs):
                        return True
                    u, v = test_sorted_pairs[idx]
                    forbidden = test_set - {u, v}
                    forbidden.update(used_interiors)
                    for max_len in [1, 2, 3, 4]:
                        path = _find_path(adj, u, v, forbidden, max_len)
                        if path:
                            new_used = set(used_interiors)
                            new_used.update(path[1:-1])
                            current_paths.append(path)
                            if find_paths_test(idx + 1, new_used, current_paths):
                                return True
                            current_paths.pop()
                    return False
                
                test_paths = []
                if find_paths_test(0, set(), test_paths):
                    final_branches = test_branches
                    found_paths = test_paths
                    break
            else:
                continue
            break
        else:
            # Ultimate fallback: return max clique with direct edges
            final_branches = list(clique)
            branch_set = set(final_branches)
            pairs = list(combinations(final_branches, 2))
            found_paths = [[u, v] for u, v in pairs]

    # Ensure types
    branches_out = [int(b) for b in final_branches]
    paths_out = [[int(p) for p in path] for path in found_paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
