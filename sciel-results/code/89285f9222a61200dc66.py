"""Rewritten program: longest simple path via matching-based seed + local search."""
import random
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # 1. Bipartite Matching Setup
    # U: 0..n-1, V: 0..n-1
    # Edge (u, v) in bipartite if (u, v) in G
    bip_adj = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            bip_adj[u].append(v)

    # Hopcroft-Karp
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in bip_adj[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == -1:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found

    def dfs(u):
        for v in bip_adj[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1

    # 2. Reconstruct Disjoint Paths
    next_node = [-1] * n
    prev_node = [-1] * n
    for u in range(n):
        if pair_u[u] != -1:
            v = pair_u[u]
            next_node[u] = v
            prev_node[v] = u

    paths = []
    visited = [False] * n
    for i in range(n):
        if not visited[i] and prev_node[i] == -1:
            path = [i]
            visited[i] = True
            cur = i
            while next_node[cur] != -1:
                cur = next_node[cur]
                path.append(cur)
                visited[cur] = True
            paths.append(path)
        
    # Handle cycles
    for i in range(n):
        if not visited[i] and prev_node[i] != -1:
            path = [i]
            visited[i] = True
            cur = i
            while next_node[cur] != -1 and not visited[next_node[cur]]:
                cur = next_node[cur]
                path.append(cur)
                visited[cur] = True
            cur = i
            while prev_node[cur] != -1 and not visited[prev_node[cur]]:
                cur = prev_node[cur]
                path.insert(0, cur)
                visited[cur] = True
            paths.append(path)

    best_path = []
    for p in paths:
        if len(p) > len(best_path):
            best_path = p

    # 3. Improve using Diameter of Component (if tree-like)
    if best_path:
        start_node = best_path[0]
        comp = set()
        q = deque([start_node])
        comp.add(start_node)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp.add(v)
                    q.append(v)
        
        edge_count = 0
        for u in comp:
            edge_count += len([v for v in adj[u] if v in comp])
        edge_count //= 2
        
        if edge_count == len(comp) - 1:
            def bfs_farthest(start):
                dist_map = {start: 0}
                parent = {start: -1}
                q = deque([start])
                farthest = start
                max_dist = 0
                while q:
                    u = q.popleft()
                    if dist_map[u] > max_dist:
                        max_dist = dist_map[u]
                        farthest = u
                    for v in adj[u]:
                        if v not in dist_map:
                            dist_map[v] = dist_map[u] + 1
                            parent[v] = u
                            q.append(v)
                path = []
                cur = farthest
                while cur != -1:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path, farthest, max_dist

            path1, mid, _ = bfs_farthest(start_node)
            path2, end, diam_len = bfs_farthest(mid)
            
            if len(path2) > len(best_path):
                best_path = path2

    # 4. Local Search: Extend ends
    def extend_path(path):
        path = list(path)
        changed = True
        while changed:
            changed = False
            if path:
                front = path[0]
                cands = [v for v in adj[front] if v not in set(path)]
                if cands:
                    best_v = None
                    best_len = 0
                    for v in cands:
                        ext = [v]
                        seen = set(path) | {v}
                        cur = v
                        while True:
                            nexts = [w for w in adj[cur] if w not in seen]
                            if not nexts:
                                break
                            w = min(nexts, key=lambda x: (len(adj[x] - seen), x))
                            ext.append(w)
                            seen.add(w)
                            cur = w
                        if len(ext) > best_len:
                            best_len = len(ext)
                            best_v = v
                    if best_v is not None:
                        path.insert(0, best_v)
                        changed = True
            
            if path:
                back = path[-1]
                cands = [v for v in adj[back] if v not in set(path)]
                if cands:
                    best_v = None
                    best_len = 0
                    for v in cands:
                        ext = [v]
                        seen = set(path) | {v}
                        cur = v
                        while True:
                            nexts = [w for w in adj[cur] if w not in seen]
                            if not nexts:
                                break
                            w = min(nexts, key=lambda x: (len(adj[x] - seen), x))
                            ext.append(w)
                            seen.add(w)
                            cur = w
                        if len(ext) > best_len:
                            best_len = len(ext)
                            best_v = v
                    if best_v is not None:
                        path.append(best_v)
                        changed = True
        return path

    best_path = extend_path(best_path)

    if not best_path:
        best_path = [0]
        
    return best_path
