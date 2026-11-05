import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
    visited = [False] * n
    components = []
    for i in range(n):
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
            components.append(comp)
    return components

def _tree_diameter_path(adj, comp):
    """Exact longest path in a tree via two BFS diameter."""
    if len(comp) <= 1:
        return list(comp)
    comp_set = set(comp)
    start = comp[0]
    def bfs(s):
        dist = {s: 0}
        par = {s: None}
        q = deque([s])
        far = s
        while q:
            u = q.popleft()
            if dist[u] > dist[far]:
                far = u
            for v in adj[u]:
                if v in comp_set and v not in dist:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
        path = []
        cur = far
        while cur is not None:
            path.append(cur)
            cur = par[cur]
        path.reverse()
        return path
    return bfs(bfs(start)[0])

def _dp_longest_path(adj, comp, deadline, max_k=22):
    """Time-bounded bitmask DP for longest path in a component of size <= max_k.
    Returns (path, full_hamiltonian_found)."""
    k = len(comp)
    if k <= 1:
        return list(comp), True
    if k > max_k:
        return None, False
    vmap = {v: i for i, v in enumerate(comp)}
    imap = {i: v for i, v in enumerate(comp)}
    local_adj = [0] * k
    for u in comp:
        ui = vmap[u]
        m = 0
        for v in adj[u]:
            if v in vmap:
                m |= 1 << vmap[v]
        local_adj[ui] = m
    full_mask = (1 << k) - 1
    # dp[mask] = bitmask of valid endpoints
    dp = [0] * (1 << k)
    for i in range(k):
        dp[1 << i] = 1 << i
    best_len = 1
    best_mask = 0
    best_end = 0
    full_found = False
    # Iterate by popcount to track the maximum achievable length exactly
    for pc in range(1, k + 1):
        if time.time() > deadline:
            break
        # masks with pc bits
        for mask in range(1, 1 << k):
            if bin(mask).count('1') != pc:
                continue
            eps = dp[mask]
            if eps == 0:
                continue
            if pc > best_len:
                best_len = pc
                best_mask = mask
                best_end = (eps & -eps).bit_length() - 1
            if pc == k:
                full_found = True
                best_mask = mask
                best_end = (eps & -eps).bit_length() - 1
                break
            # extend
            for last in range(k):
                if not (eps & (1 << last)):
                    continue
                avail = local_adj[last] & ~mask
                while avail:
                    lb = avail & -avail
                    nxt = lb.bit_length() - 1
                    dp[mask | lb] |= lb
                    avail ^= lb
        if full_found:
            break
    # reconstruct
    path = []
    mask = best_mask
    end = best_end
    while mask:
        path.append(end)
        if mask == (1 << end):
            break
        prev_mask = mask ^ (1 << end)
        found = False
        eps = dp[prev_mask]
        while eps:
            lb = eps & -eps
            prev = lb.bit_length() - 1
            if local_adj[prev] & (1 << end):
                path.append(prev)
                mask = prev_mask
                end = prev
                found = True
                break
            eps ^= lb
        if not found:
            break
    path.reverse()
    result = [imap[v] for v in path]
    return result, full_found

def _greedy_path(adj, start, visited):
    path = [start]
    visited = set(visited)
    visited.add(start)
    while True:
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        if not cands:
            break
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        nxt = cands[0]
        path.append(nxt)
        visited.add(nxt)
    return path

def _try_extend_path(adj, path, n, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        path_set = set(path)
        end = path[-1]
        cands = [nb for nb in adj[end] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.append(cands[0])
            improved = True
            continue
        start = path[0]
        cands = [nb for nb in adj[start] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.insert(0, cands[0])
            improved = True
    return path

def _insert_vertex(adj, path, vertex):
    if vertex in set(path):
        return None
    if vertex in adj[path[0]]:
        return [vertex] + path
    if vertex in adj[path[-1]]:
        return path + [vertex]
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            return path[:i+1] + [vertex] + path[i+1:]
    return None

def _limited_dfs(adj, start, deadline, max_depth):
    best_path = [start]
    best_len = 1
    def dfs(path, visited):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        if len(path) >= max_depth:
            return
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        for nxt in cands:
            if time.time() > deadline:
                return
            path.append(nxt)
            visited.add(nxt)
            dfs(path, visited)
            path.pop()
            visited.remove(nxt)
    dfs([start], {start})
    return best_path

def construct():
    n = P['n']
    adj = _build_adj()
    deadline = time.time() + 80.0

    components = _find_components(adj, n)
    components.sort(key=len, reverse=True)

    best_path = []
    best_len = 0

    for comp in components:
        if time.time() > deadline - 3.0:
            break
        comp_size = len(comp)
        if comp_size <= best_len:
            continue
        edge_count = sum(len(adj[u]) for u in comp) // 2
        if edge_count == comp_size - 1:
            path = _tree_diameter_path(adj, comp)
            if len(path) > best_len:
                best_path = path
                best_len = len(path)
            continue
        if comp_size <= 22:
            path, full = _dp_longest_path(adj, comp, deadline)
            if path is not None and len(path) > best_len:
                best_path = path
                best_len = len(path)
        else:
            # heuristic for large non-tree component
            for start in comp:
                if time.time() > deadline - 3.0:
                    break
                p = _greedy_path(adj, start, set())
                p = _try_extend_path(adj, p, n, deadline)
                if len(p) > best_len:
                    best_path = p
                    best_len = len(p)

    # If the main component was solved fully (Hamiltonian), the path is optimal
    # among components of that size; still try to extend in case of multi-component
    best_path = _try_extend_path(adj, best_path, n, deadline)

    # Refinement: insert any vertex not in path
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        path_set = set(best_path)
        for v in range(n):
            if v not in path_set:
                np = _insert_vertex(adj, best_path, v)
                if np is not None:
                    best_path = np
                    path_set = set(best_path)
                    improved = True
                    break
    best_path = _try_extend_path(adj, best_path, n, deadline)

    # If we have spare time and haven't reached full component, do a few DFS probes
    if time.time() < deadline - 5.0 and len(best_path) < 60:
        rng = random.Random(748)
        starts = list(range(n))
        rng.shuffle(starts)
        for start in starts[:30]:
            if time.time() > deadline - 3.0:
                break
            p = _limited_dfs(adj, start, deadline - 3.0, max_depth=30)
            p = _try_extend_path(adj, p, n, deadline)
            if len(p) > len(best_path):
                best_path = p

    # Final verification
    if len(best_path) == 0:
        best_path = [0]
    for i in range(len(best_path) - 1):
        assert best_path[i+1] in adj[best_path[i]], f"Invalid edge at position {i}"
    assert len(set(best_path)) == len(best_path), "Path has repeated vertices"
    return best_path
