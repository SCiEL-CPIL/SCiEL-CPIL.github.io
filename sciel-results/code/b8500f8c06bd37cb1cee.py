import random
import time
import sys
from collections import deque

def construct():
    N = 100
    p = 1.8 / 100
    rng = random.Random(748)

    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components
    visited = [False] * N
    components = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)

    components.sort(key=len, reverse=True)
    if not components or not components[0]:
        return [0]

    largest_comp = components[0]
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)

    start_time = time.time()
    deadline = start_time + 84.0   # safety margin before 90s kill

    def time_ok():
        return time.time() < deadline

    def in_comp(v):
        return v in comp_set

    # ---------- Tree case: exact diameter ----------
    edges_in_comp = sum(len(adj[u] & comp_set) for u in largest_comp) // 2
    is_tree = (edges_in_comp == comp_size - 1)

    def bfs_path(start_node):
        dist = {start_node: 0}
        parent = {start_node: None}
        queue = deque([start_node])
        farthest = start_node
        max_dist = 0
        while queue:
            u = queue.popleft()
            if dist[u] > max_dist:
                max_dist = dist[u]
                farthest = u
            for v in adj[u]:
                if v in comp_set and v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        path = []
        curr = farthest
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path, max_dist

    if is_tree:
        path1, _ = bfs_path(largest_comp[0])
        path2, _ = bfs_path(path1[-1])
        return path2

    # ---------- Cyclic case: greedy + refinement + beam + DFS ----------
    best_path = []
    best_len = 0

    def update(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            if best_len == comp_size:
                return True
        return False

    def unvisited_degree(n, seen):
        return sum(1 for x in adj[n] if in_comp(x) and x not in seen)

    # ---- Greedy: forced-move + ascending-degree tie-break + random tie-break ----
    def greedy(start, rng_local):
        seen = {start}
        path = [start]
        cur = start
        while True:
            if not time_ok():
                break
            nbrs = [n for n in adj[cur] if in_comp(n) and n not in seen]
            if not nbrs:
                break
            if len(nbrs) == 1:
                nxt = nbrs[0]
            else:
                nbrs.sort(key=lambda n: (unvisited_degree(n, seen), rng_local.random()))
                nxt = nbrs[0]
            seen.add(nxt)
            path.append(nxt)
            cur = nxt
        return path

    # ---- Extend both ends by greedily walking into unvisited neighbors ----
    def extend_both(path):
        path = list(path)
        seen = set(path)
        changed = True
        while changed and time_ok():
            changed = False
            # extend left
            while time_ok():
                cur = path[0]
                nbrs = [n for n in adj[cur] if in_comp(n) and n not in seen]
                if not nbrs:
                    break
                nbrs.sort(key=lambda n: (unvisited_degree(n, seen), random.random()))
                nxt = nbrs[0]
                path.insert(0, nxt)
                seen.add(nxt)
                changed = True
            # extend right
            while time_ok():
                cur = path[-1]
                nbrs = [n for n in adj[cur] if in_comp(n) and n not in seen]
                if not nbrs:
                    break
                nbrs.sort(key=lambda n: (unvisited_degree(n, seen), random.random()))
                nxt = nbrs[0]
                path.append(nxt)
                seen.add(nxt)
                changed = True
        return path

    # ---- 2-opt refinement (reverses a subpath, preserves simple path) ----
    def two_opt(path):
        path = list(path)
        n = len(path)
        improved = True
        iters = 0
        while improved and time_ok() and iters < 200:
            improved = False
            iters += 1
            for i in range(1, n - 1):
                if not time_ok():
                    break
                for j in range(i + 1, n):
                    if not time_ok():
                        break
                    # reverse path[i..j]; check new edges (i-1,i) and (j,j+1)
                    a = path[i - 1]
                    b = path[i]
                    c = path[j]
                    d = path[j + 1] if j + 1 < n else None
                    # after reversing i..j: ... a, c, ..., b, d ...
                    if a in adj[c] and (d is None or b in adj[d]):
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True
                        break
                if improved:
                    break
        return path

    # ---- Insertion refinement: move an unvisited node into the path ----
    def insert_refine(path):
        path = list(path)
        seen = set(path)
        unvisited = [v for v in largest_comp if v not in seen]
        improved = True
        iters = 0
        while improved and time_ok() and iters < 200:
            improved = False
            iters += 1
            for v in unvisited:
                if not time_ok():
                    break
                # try inserting v between path[k] and path[k+1]
                best_k = -1
                for k in range(len(path) - 1):
                    if path[k] in adj[v] and path[k + 1] in adj[v]:
                        best_k = k
                        break
                if best_k >= 0:
                    path.insert(best_k + 1, v)
                    seen.add(v)
                    unvisited.remove(v)
                    improved = True
                    break
        return path

    def refine(path):
        path = extend_both(path)
        path = two_opt(path)
        path = insert_refine(path)
        path = two_opt(path)
        path = extend_both(path)
        return path

    # ---- Beam search (bounded width/depth) ----
    def beam_search(start, width=6, max_depth=40):
        # state: (path_tuple, current, seen_frozenset)
        start_state = ((start,), start, frozenset({start}))
        best = [start_state]
        for _ in range(max_depth):
            if not time_ok():
                break
            cands = []
            for (ptup, cur, seen) in best:
                nbrs = [n for n in adj[cur] if in_comp(n) and n not in seen]
                if not nbrs:
                    continue
                nbrs.sort(key=lambda n: (unvisited_degree(n, set(seen)), random.random()))
                for nxt in nbrs[:max(1, width // 2)]:
                    cands.append((ptup + (nxt,), nxt, seen | {nxt}))
            if not cands:
                break
            # keep top by length, then by sum of remaining degree (more options first)
            def score(c):
                ptup, cur, seen = c
                rem = sum(1 for x in adj[cur] if in_comp(x) and x not in seen)
                return (len(ptup), rem)
            cands.sort(key=score, reverse=True)
            best = cands[:width]
            # track best length
            for (ptup, cur, seen) in best:
                if len(ptup) > best_len:
                    best_len = len(ptup)
                    best_path = list(ptup)
            if best_len == comp_size:
                break
        # return longest beam state
        best_state = max(best, key=lambda c: len(c[0])) if best else None
        if best_state is not None:
            return list(best_state[0])
        return list(start)

    # ---- Bounded DFS with Warnsdorff ordering (per start) ----
    def dfs_bounded(start, node_budget=200000):
        nonlocal best_path, best_len
        seen = {start}
        path = [start]
        budget = [node_budget]
        found_ham = [False]

        sys.setrecursionlimit(100000)

        def dfs(cur):
            if found_ham[0] or not time_ok() or budget[0] <= 0:
                return
            budget[0] -= 1
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                if best_len == comp_size:
                    found_ham[0] = True
                    return
            nbrs = [n for n in adj[cur] if in_comp(n) and n not in seen]
            nbrs.sort(key=lambda n: (unvisited_degree(n, seen), random.random()))
            for nxt in nbrs:
                if found_ham[0] or not time_ok() or budget[0] <= 0:
                    return
                seen.add(nxt)
                path.append(nxt)
                dfs(nxt)
                path.pop()
                seen.remove(nxt)

        dfs(start)
        return list(best_path)

    # ---- Order starts: low degree first (leaves/dead-ends), then all ----
    starts = sorted(largest_comp, key=lambda n: (len(adj[n] & comp_set), n))
    # interleave: low-degree first, then higher
    order = list(starts)
    # append high-degree ones too (already in starts sorted ascending, so high-degree last)

    # Phase 1: greedy + refine over many starts (cheap, many restarts)
    rng_local = random.Random(12345)
    for s in order:
        if not time_ok():
            break
        gp = greedy(s, rng_local)
        rp = refine(gp)
        update(rp)
        if best_len == comp_size:
            break

    if best_len == comp_size:
        return best_path

    # Phase 2: beam search over promising starts
    for s in order[:30]:
        if not time_ok():
            break
        bp = beam_search(s)
        rp = refine(bp)
        update(rp)
        if best_len == comp_size:
            break

    if best_len == comp_size:
        return best_path

    # Phase 3: bounded DFS over a few low-degree starts
    for s in order[:8]:
        if not time_ok():
            break
        dfs_bounded(s, node_budget=150000)
        if best_len == comp_size:
            break

    # Final polish on best_path
    if best_path:
        final = refine(best_path)
        update(final)

    if not best_path:
        best_path = [largest_comp[0]]

    return best_path
