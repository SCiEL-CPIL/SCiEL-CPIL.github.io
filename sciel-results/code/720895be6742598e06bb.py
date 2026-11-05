"""Improved long-path finder: tree-component diameters + branch-and-bound search."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Find connected components
def get_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

# BFS to find farthest node and distances
def bfs_farthest(start, adj, n):
    dist = [-1] * n
    dist[start] = 0
    queue = [start]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    farthest = max(range(n), key=lambda x: dist[x] if dist[x] >= 0 else -1)
    return farthest, dist

# Find tree diameter (longest path in a tree)
def tree_diameter(comp, adj, n):
    if len(comp) <= 1:
        return comp
    # BFS from arbitrary node to find farthest
    start = comp[0]
    far1, _ = bfs_farthest(start, adj, n)
    far2, dist = bfs_farthest(far1, adj, n)
    # Reconstruct path from far1 to far2
    # BFS from far2 to get parent pointers
    parent = [-1] * n
    dist2 = [-1] * n
    dist2[far2] = 0
    queue = [far2]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                parent[v] = u
                queue.append(v)
    # Walk back from far1 to far2
    path = []
    cur = far1
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

# Branch and bound for longest path in a general graph
def longest_path_bnb(adj, n, deadline, best_len):
    """Branch and bound to find longest simple path."""
    best_path = []
    
    def dfs(path, visited):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = list(path)
        # Upper bound: remaining unvisited nodes
        if len(path) + (n - len(visited)) <= len(best_path):
            return
        # Try extensions
        last = path[-1]
        candidates = [v for v in adj[last] if v not in visited]
        # Sort by degree of unvisited neighbors (fewer first - heuristic)
        candidates.sort(key=lambda v: sum(1 for w in adj[v] if w not in visited))
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            visited.add(v)
            dfs(path, visited)
            path.pop()
            visited.discard(v)
    
    # Start from each vertex
    for start in range(n):
        if time.time() > deadline:
            break
        dfs([start], {start})
    
    return best_path

# Local search: try to extend a path from both ends
def try_extend(path, adj, n):
    """Try to extend path from both ends."""
    changed = True
    while changed:
        changed = False
        # Try extending from the end
        last = path[-1]
        for v in adj[last]:
            if v not in set(path):
                path.append(v)
                changed = True
                break
        # Try extending from the start
        first = path[0]
        for v in adj[first]:
            if v not in set(path):
                path.insert(0, v)
                changed = True
                break
    return path

# 2-opt style local improvement: try swapping/reordering
def local_improve(path, adj, n, deadline):
    """Try to find a longer path by local modifications."""
    if not path:
        return path
    best = list(path)
    best_set = set(best)
    
    # Try inserting unvisited vertices into the path
    unvisited = [v for v in range(n) if v not in best_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                # Check if we can insert an unvisited vertex between best[i] and best[j]
                # Actually, for a path, consecutive elements must be adjacent
                # So we can only insert between consecutive elements
                pass
        
        # Simpler: try to insert each unvisited vertex at each position
        for v in unvisited[:]:
            if time.time() > deadline:
                break
            for pos in range(len(best) + 1):
                if time.time() > deadline:
                    break
                # Check if insertion is valid
                valid = True
                if pos > 0 and best[pos - 1] not in adj[v]:
                    valid = False
                if valid and pos < len(best) and best[pos] not in adj[v]:
                    valid = False
                if valid:
                    new_path = best[:pos] + [v] + best[pos:]
                    # Verify all consecutive pairs are edges
                    ok = True
                    for k in range(len(new_path) - 1):
                        if new_path[k + 1] not in adj[new_path[k]]:
                            ok = False
                            break
                    if ok:
                        best = new_path
                        best_set.add(v)
                        unvisited.remove(v)
                        improved = True
                        break
            if improved:
                break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # Step 1: Find connected components
    comps = get_components()
    
    # Step 2: For each component, find the longest path
    best_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_set = set(comp)
        comp_size = len(comp)
        
        if comp_size <= 1:
            if comp_size > len(best_path):
                best_path = list(comp)
            continue
        
        # Check if component is a tree (edges = vertices - 1)
        edge_count = sum(len(adj[v] & comp_set) for v in comp) // 2
        is_tree = (edge_count == comp_size - 1)
        
        if is_tree:
            # Use tree diameter
            path = tree_diameter(comp, adj, N)
            if len(path) > len(best_path):
                best_path = path
        else:
            # Use branch and bound
            path = longest_path_bnb(adj, N, deadline, len(best_path))
            if len(path) > len(best_path):
                best_path = path
    
    # Step 3: Local improvement on the best path found
    if time.time() < deadline - 5:
        best_path = try_extend(best_path, adj, N)
    
    if time.time() < deadline - 3:
        best_path = local_improve(best_path, adj, N, deadline - 3)
    
    # Step 4: Try additional BnB runs with different starting strategies if time remains
    if time.time() < deadline - 5 and len(best_path) < N:
        # Try BnB with a tighter deadline for refinement
        path2 = longest_path_bnb(adj, N, deadline - 2, len(best_path))
        if len(path2) > len(best_path):
            best_path = path2
            best_path = try_extend(best_path, adj, N)
    
    # Ensure we return a nonempty valid path
    if not best_path:
        best_path = [0]
    
    # Final validation
    best_set = set(best_path)
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # Truncate at first invalid point
            best_path = best_path[:i + 1]
            break
    
    return best_path
