import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_path(u, v, banned):
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def find_shortest_path(u, v, banned):
        """BFS level-by-level to guarantee shortest path (fewest interior vertices)."""
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            level_size = len(q)
            for _ in range(level_size):
                cur = q.popleft()
                for nxt in adj[cur]:
                    if nxt == v:
                        path = [v]
                        node = cur
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    if nxt in parent or nxt in banned:
                        continue
                    parent[nxt] = cur
                    q.append(nxt)
        return None

    def try_find(branches, rng, attempts=60):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort pairs by "difficulty": prefer pairs that are NOT directly adjacent
        # (those need longer paths and more interior vertices)
        def pair_key(p):
            a, b = p
            if b in adj[a]:
                return (0, degrees[a] + degrees[b])
            else:
                return (1, -(degrees[a] + degrees[b]))
        deg_order = sorted(pairs, key=pair_key)

        for attempt in range(attempts):
            if time_up():
                break
            if attempt == 0:
                order = list(deg_order)
            elif attempt % 4 == 1:
                order = list(deg_order)
                order.reverse()
            elif attempt % 4 == 2:
                order = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])
            else:
                order = list(pairs)
                rng.shuffle(order)

            used = set()
            paths = {}
            ok = True
            for a, b in order:
                banned = used | bset
                banned.discard(a)
                banned.discard(b)
                p = find_shortest_path(a, b, banned)
                if p is None:
                    ok = False
                    break
                interior = set(p[1:-1])
                if interior & bset or interior & used:
                    ok = False
                    break
                used |= interior
                paths[frozenset((a, b))] = p
            if ok:
                return paths
        return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected):
            return 0
        used = set()
        for fkey, p in paths.items():
            if len(p) < 2:
                return 0
            a, b = p[0], p[-1]
            if a not in bset or b not in bset or a == b:
                return 0
            if fkey not in expected:
                return 0
            interior = set(p[1:-1])
            if interior & bset or interior & used:
                return 0
            for i in range(len(p) - 1):
                if p[i+1] not in adj[p[i]]:
                    return 0
            used |= interior
        return len(branches)

    def final_result(branches, paths):
        branch_list = [int(b) for b in sorted(branches)]
        path_list = []
        for a, b in combinations(sorted(branches), 2):
            fkey = frozenset((a, b))
            p = paths.get(fkey)
            if p is None:
                p = [a, b]
            path_list.append([int(x) for x in p])
        return {'branches': branch_list, 'paths': path_list}

    rng = random.Random(42)
    best_count = 0
    best_result = None

    # Phase 1: Top-down k search
    for k in range(14, 1, -1):
        if time_up():
            break
        candidates = []
        seen_sets = set()
        
        # Top k by degree
        c = tuple(sorted(sorted_verts[:k]))
        candidates.append(sorted_verts[:k])
        seen_sets.add(frozenset(c))
        
        # Top k-1 + random (more samples)
        for _ in range(8):
            top_k1 = sorted_verts[:k-1]
            rest = [v for v in range(n) if v not in top_k1]
            if rest:
                c = sorted(top_k1 + [rng.choice(rest)])
                fs = frozenset(c)
                if fs not in seen_sets:
                    candidates.append(c)
                    seen_sets.add(fs)
        
        # Random (more samples)
        for _ in range(30):
            c = sorted(rng.sample(range(n), k))
            fs = frozenset(c)
            if fs not in seen_sets:
                candidates.append(c)
                seen_sets.add(fs)
        
        # Medium-degree mix: top half + bottom half
        for _ in range(4):
            half = k // 2
            c1 = sorted_verts[:half]
            c2 = sorted_verts[n//2:n//2 + (k - half)]
            c = sorted(c1 + c2)
            fs = frozenset(c)
            if fs not in seen_sets:
                candidates.append(c)
                seen_sets.add(fs)

        for cand in candidates:
            if time_up():
                break
            paths = try_find(cand, rng, attempts=80)
            if paths is not None:
                cnt = validate(cand, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (cand, paths)
                    break
        if best_result is not None:
            break

    # Phase 2: Greedy growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        current_paths = try_find(current, rng, attempts=80)
        for _ in range(15):
            if time_up():
                break
            if current_paths is None:
                break
            cset = set(current)
            cands = [v for v in range(n) if v not in cset]
            cands.sort(key=lambda v: degrees[v], reverse=True)
            added = False
            for v in cands:
                if time_up():
                    break
                new_br = current + [v]
                new_paths = try_find(new_br, rng, attempts=50)
                if new_paths is not None:
                    cnt = validate(new_br, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_br, new_paths)
                        current = new_br
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break

    # Fallback
    if best_result is None:
        # Try a few more quick attempts with shortest path
        for k in range(4, 1, -1):
            if time_up():
                break
            cand = sorted_verts[:k]
            paths = try_find(cand, rng, attempts=20)
            if paths is not None:
                best_result = (cand, paths)
                break
        if best_result is None:
            u, v = edges[0]
            best_result = ([u, v], {frozenset((u, v)): [u, v]})

    branches, paths = best_result
    return final_result(branches, paths)
