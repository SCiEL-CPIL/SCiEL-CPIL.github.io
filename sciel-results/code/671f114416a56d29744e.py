import time
import heapq
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return adj

ADJ = build_adj()
DEG = [len(a) for a in ADJ]

def find_components(adj, n):
    """Find all connected components."""
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)
    return components

def greedy_path(start, adj, n, strategy='warnsdorff'):
    """Grow a path greedily from start vertex."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'warnsdorff':
            # Pick neighbor with minimum remaining degree
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_degree':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
        else:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, adj, n, deadline=None):
    """Try to extend a path from both ends repeatedly."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        if deadline and time.time() > deadline:
            break
        changed = False
        # Try extending from start
        cur = path[0]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Try extending from end
        cur = path[-1]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def local_search_2opt(path, adj, n, deadline=None, max_iter=200):
    """Try to improve path length via 2-opt style moves: 
    If path is ...a-b-c... and there exist edges a-c and b-d (or d-b) for some 
    unvisited d, we can restructure. More practically, try swapping internal 
    segments or finding shortcuts that allow extension."""
    if len(path) < 3:
        return path
    
    best_path = list(path)
    best_len = len(best_path)
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try: for each pair of consecutive edges (i-1, i) and (i, i+1),
        # see if we can replace vertex i with another that connects to both,
        # freeing up the original vertex for extension.
        for i in range(1, len(best_path) - 1):
            prev_v = best_path[i-1]
            next_v = best_path[i+1]
            cur_v = best_path[i]
            
            # Find vertices that connect to both prev_v and next_v
            common = (ADJ[prev_v] & ADJ[next_v]) - set(best_path)
            for cand in common:
                if cand == cur_v:
                    continue
                # Replace cur_v with cand
                new_path = list(best_path)
                new_path[i] = cand
                # Now cur_v is free; try to extend
                extended = extend_both_ends(new_path, adj, n, deadline)
                if len(extended) > best_len:
                    best_path = extended
                    best_len = len(best_path)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_path

def splice_paths(path1, path2, adj, n):
    """Try to splice two disjoint paths into one longer path.
    If path1 = [a1, a2, ..., ak] and path2 = [b1, b2, ..., bm],
    we can splice if ak is adjacent to b1 (or bm) and the paths are disjoint,
    giving [a1,...,ak,b1,...,bm] or [a1,...,ak,bm,...,b1]."""
    if not path1 or not path2:
        return path1 if path1 else path2
    
    set1 = set(path1)
    set2 = set(path2)
    if set1 & set2:
        return path1 if len(path1) >= len(path2) else path2
    
    # Try: end of path1 connects to start of path2
    if path1[-1] in ADJ[path2[0]]:
        merged = path1 + path2
        return extend_both_ends(merged, adj, n)
    
    # Try: end of path1 connects to end of path2
    if path1[-1] in ADJ[path2[-1]]:
        merged = path1 + path2[::-1]
        return extend_both_ends(merged, adj, n)
    
    # Try: start of path1 connects to start of path2
    if path1[0] in ADJ[path2[0]]:
        merged = path1[::-1] + path2
        return extend_both_ends(merged, adj, n)
    
    # Try: start of path1 connects to end of path2
    if path1[0] in ADJ[path2[-1]]:
        merged = path1 + path2[::-1]
        return extend_both_ends(merged, adj, n)
    
    # Try all endpoint combinations
    for i in range(len(path1)):
        for j in range(len(path2)):
            if path1[i] in ADJ[path2[j]]:
                # Can we restructure? This is complex; skip for now
                pass
    
    return path1 if len(path1) >= len(path2) else path2

def systematic_dfs(start, adj, n, deadline, target_len):
    """Systematic DFS trying to find a path of at least target_len.
    Uses Warnsdorff's rule for ordering."""
    best = []
    
    def dfs(path, seen):
        nonlocal best
        if time.time() > deadline:
            return
        if len(path) > len(best):
            best = list(path)
        if len(path) >= target_len:
            return
        
        cur = path[-1]
        candidates = list(adj[cur] - seen)
        if not candidates:
            return
        
        # Sort by Warnsdorff: minimum remaining degree first
        candidates.sort(key=lambda w: (len(adj[w] - seen), w))
        
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if len(path) >= target_len:
                return
    
    dfs([start], {start})
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    # Step 1: Find components
    components = find_components(ADJ, N)
    # Sort by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    
    # Focus on the largest component
    if components:
        main_comp = components[0]
        main_set = set(main_comp)
        comp_size = len(main_comp)
    else:
        return [0]
    
    # Step 2: Greedy from all vertices in main component with multiple strategies
    for start in main_comp:
        if time.time() > deadline:
            break
        for strategy in ['warnsdorff', 'max_degree']:
            path = greedy_path(start, ADJ, N, strategy)
            path = extend_both_ends(path, ADJ, N, deadline)
            if len(path) > len(best_path):
                best_path = list(path)
    
    # Step 3: Systematic DFS from promising starts
    if len(best_path) < comp_size and time.time() < deadline - 10:
        # Try DFS from each vertex in the component
        target = len(best_path) + 1
        for start in main_comp:
            if time.time() > deadline - 5:
                break
            # Only try if degree > 0
            if DEG[start] == 0:
                continue
            path = systematic_dfs(start, ADJ, N, deadline - 3, target)
            if path and len(path) > len(best_path):
                best_path = list(path)
                target = len(best_path) + 1
                if len(best_path) >= comp_size:
                    break
    
    # Step 4: Local search improvement
    if len(best_path) > 0 and time.time() < deadline - 5:
        improved = local_search_2opt(best_path, ADJ, N, deadline - 3, max_iter=100)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Step 5: Generate multiple paths and try to splice them
    if len(best_path) < comp_size and time.time() < deadline - 10:
        # Generate several paths from different starts
        paths = []
        for start in main_comp[:30]:  # First 30 vertices
            if time.time() > deadline - 8:
                break
            path = greedy_path(start, ADJ, N, 'warnsdorff')
            path = extend_both_ends(path, ADJ, N, deadline)
            if len(path) > 2:
                paths.append(path)
        
        # Try to splice pairs of disjoint paths
        for i in range(len(paths)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(paths)):
                if time.time() > deadline - 5:
                    break
                set_i = set(paths[i])
                set_j = set(paths[j])
                if set_i & set_j:
                    continue
                merged = splice_paths(paths[i], paths[j], ADJ, N)
                if merged and len(merged) > len(best_path):
                    best_path = list(merged)
        
        # Also try to merge small paths into the best path
        for path in paths:
            if time.time() > deadline - 5:
                break
            if len(path) < 3:
                continue
            set_best = set(best_path)
            set_path = set(path)
            if set_best & set_path:
                continue
            merged = splice_paths(best_path, path, ADJ, N)
            if merged and len(merged) > len(best_path):
                best_path = list(merged)
    
    # Step 6: Randomized restarts
    if len(best_path) < comp_size and time.time() < deadline - 3:
        rng = random.Random(748)
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            start = rng.choice(main_comp)
            path = greedy_path(start, ADJ, N, 'warnsdorff')
            path = extend_both_ends(path, ADJ, N, deadline)
            if len(path) > len(best_path):
                best_path = list(path)
    
    # Step 7: Final extension attempt
    if len(best_path) < comp_size:
        extended = extend_both_ends(best_path, ADJ, N, deadline)
        if extended and len(extended) > len(best_path):
            best_path = extended
    
    # Step 8: One more local search pass
    if len(best_path) > 0 and time.time() < deadline - 1:
        improved = local_search_2opt(best_path, ADJ, N, deadline, max_iter=50)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Ensure non-empty
    if not best_path:
        best_path = [main_comp[0]]
    
    return best_path
# EVOLVE-BLOCK-END
