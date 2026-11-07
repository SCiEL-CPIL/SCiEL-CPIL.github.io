"""Rewritten program for clique subdivision in a pinned graph."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Vertex ordering by degree (descending)
    vertex_order = sorted(range(n), key=lambda x: -degrees[x])
    
    def time_up():
        return time.time() - start_time > DEADLINE
    
    def find_path(src, dst, blocked, branches_set):
        """Find a simple path from src to dst avoiding blocked interior vertices."""
        # BFS for shortest path
        if src == dst:
            return None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == dst:
                    # Reconstruct path
                    path = [dst]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    # Check that interior vertices are not blocked
                    interior = path[1:-1]
                    for x in interior:
                        if x in blocked or x in branches_set:
                            return None
                    return path
                if v not in visited and v not in blocked and v not in branches_set:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        return None
    
    def find_path_bfs(src, dst, blocked, branches_set):
        """BFS shortest path avoiding blocked and branch vertices in interior."""
        if src == dst:
            return [src, dst] if src in adj[dst] else None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == dst:
                    # Reconstruct path
                    path = [dst]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    interior = path[1:-1]
                    for x in interior:
                        if x in blocked or x in branches_set:
                            return None
                    return path
                if v not in visited and v not in blocked and v not in branches_set:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        return None
    
    def try_route_paths(branches):
        """Try to route all pairwise paths for the given branch set.
        Returns dict of (i,j) -> path, or None if fails."""
        k = len(branches)
        branches_set = set(branches)
        blocked = set()
        paths = {}
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((i, j))
        
        # Sort pairs by "difficulty" - shorter expected paths first
        # Use BFS distance as heuristic
        def pair_difficulty(pair):
            i, j = pair
            src, dst = branches[i], branches[j]
            # BFS distance ignoring blocked
            if src in adj[dst]:
                return 0
            # Quick BFS
            visited = {src}
            queue = deque([(src, 0)])
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v == dst:
                        return d + 1
                    if v not in visited and v not in branches_set:
                        visited.add(v)
                        queue.append((v, d+1))
            return 999
        
        pairs.sort(key=pair_difficulty)
        
        # Greedy: assign paths in order
        for (i, j) in pairs:
            src, dst = branches[i], branches[j]
            path = find_path_bfs(src, dst, blocked, branches_set)
            if path is None:
                return None
            interior = path[1:-1]
            for x in interior:
                if x in blocked:
                    return None
                blocked.add(x)
            paths[(i, j)] = path
        
        return paths
    
    def try_route_paths_backtrack(branches):
        """Try to route paths with backtracking on path choices."""
        k = len(branches)
        branches_set = set(branches)
        
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((i, j))
        
        # Order pairs by difficulty
        def pair_difficulty(pair):
            i, j = pair
            src, dst = branches[i], branches[j]
            if src in adj[dst]:
                return 0
            visited = {src}
            queue = deque([(src, 0)])
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v == dst:
                        return d + 1
                    if v not in visited and v not in branches_set:
                        visited.add(v)
                        queue.append((v, d+1))
            return 999
        
        pairs.sort(key=pair_difficulty)
        
        def backtrack(idx, blocked, paths):
            if time_up():
                return None
            if idx == len(pairs):
                return dict(paths)
            i, j = pairs[idx]
            src, dst = branches[i], branches[j]
            # Find all simple paths (limited)
            path = find_path_bfs(src, dst, blocked, branches_set)
            if path is None:
                return None
            interior = path[1:-1]
            for x in interior:
                blocked.add(x)
            paths[(i, j)] = path
            result = backtrack(idx + 1, blocked, paths)
            if result is not None:
                return result
            # Backtrack
            for x in interior:
                blocked.discard(x)
            del paths[(i, j)]
            return None
        
        return backtrack(0, set(), {})
    
    # Greedy construction: start with high-degree vertices, add more if routable
    best_branches = []
    best_paths = None
    
    # Try different starting sets
    # Strategy: try all subsets of top-m vertices for various m
    
    # First, get a baseline: maximum clique (direct edges)
    # Use Bron-Kerbosch for maximum clique
    adj_bits = [0] * n
    for u, v in P['edges']:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        U = P_set | X
        if not U:
            return
        pivot = -1
        max_common = -1
        temp_U = U
        while temp_U:
            lsb = temp_U & -temp_U
            u = lsb.bit_length() - 1
            common = P_set & adj_bits[u]
            cnt = bin(common).count('1')
            if cnt > max_common:
                max_common = cnt
                pivot = u
            temp_U ^= lsb
        candidates = P_set & ~adj_bits[pivot]
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            bronk(R + [v], P_set & adj_bits[v], X & adj_bits[v])
            P_set ^= lsb
            X |= lsb
            c ^= lsb
    
    bronk([], (1 << n) - 1, 0)
    
    if best_clique:
        bc = [int(x) for x in best_clique]
        if len(bc) >= 2:
            best_branches = bc[:]
            best_paths = {}
            for i in range(len(bc)):
                for j in range(i+1, len(bc)):
                    best_paths[(i,j)] = [bc[i], bc[j]]
    
    # Now try to do better with subdivision paths
    # Try increasing k from len(best_clique)+1 upward
    
    # Candidate pool: top vertices by degree
    top_vertices = vertex_order[:16]  # consider top 16
    
    # Try k from high to low
    for k in range(12, max(2, len(best_clique)), -1):
        if time_up():
            break
        found = False
        # Try combinations from top vertices
        # Use a smarter approach: greedy with backtracking
        # For now, try some random-ish subsets
        import random
        random.seed(42)
        
        # Try greedy construction
        # Start with a seed set, try to expand
        for attempt in range(200):
            if time_up():
                break
            # Random subset of top vertices of size k
            if attempt < 50:
                # Systematic: take first k from top
                cand = top_vertices[:k]
            elif attempt < 100:
                # Random sample
                cand = random.sample(top_vertices, k)
            else:
                # Mix
                cand = random.sample(range(n), k)
            
            if len(cand) < k:
                continue
            
            # Try to route
            paths = try_route_paths(cand)
            if paths is not None:
                if k > len(best_branches):
                    best_branches = cand[:]
                    best_paths = paths
                    found = True
                    # Continue trying to find even larger
                    break
            elif k <= len(best_clique) + 2:
                # Try backtracking for smaller k
                paths = try_route_paths_backtrack(cand)
                if paths is not None:
                    if k > len(best_branches):
                        best_branches = cand[:]
                        best_paths = paths
                        found = True
                        break
        
        if found and k > len(best_clique):
            # Found a subdivision larger than clique, continue to try larger
            pass
        elif not found and k <= len(best_clique) + 1:
            # No improvement possible at this k, and we've gone below clique size
            break
    
    # Also try greedy incremental: start with 2 branches, add one at a time
    if not time_up():
        # Greedy: start with top 2, try to add
        current = [vertex_order[0], vertex_order[1]]
        for v in vertex_order[2:]:
            if time_up():
                break
            trial = current + [v]
            paths = try_route_paths(trial)
            if paths is not None:
                current = trial
                if len(current) > len(best_branches):
                    best_branches = current[:]
                    best_paths = paths
            else:
                # Try backtracking
                if len(trial) <= 10:
                    paths = try_route_paths_backtrack(trial)
                    if paths is not None:
                        current = trial
                        if len(current) > len(best_branches):
                            best_branches = current[:]
                            best_paths = paths
    
    # Final: ensure we have at least 2 branches
    if len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(0,1): [u, v]}
    
    # Build output
    branches = [int(x) for x in best_branches]
    k = len(branches)
    
    # Reconstruct paths list in order
    paths_list = []
    for i in range(k):
        for j in range(i+1, k):
            if best_paths is not None and (i,j) in best_paths:
                p = [int(x) for x in best_paths[(i,j)]]
            elif i in range(k) and j in range(k):
                # Should not happen
                p = [branches[i], branches[j]]
            else:
                p = [branches[i], branches[j]]
            paths_list.append(p)
    
    return {'branches': branches, 'paths': paths_list}

# EVOLVE-BLOCK-END
