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

    def find_path_bfs(u, v, banned):
        """BFS for shortest path avoiding banned vertices."""
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in sorted(adj[cur]):  # deterministic order
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

    def try_find_paths(branches, pair_order):
        """Try to find disjoint paths for given branches in given pair order."""
        bset = set(branches)
        used = set()
        paths = {}
        for a, b in pair_order:
            banned = used | bset
            banned.discard(a)
            banned.discard(b)
            p = find_path_bfs(a, b, banned)
            if p is None:
                return None
            interior = set(p[1:-1])
            if interior & bset or interior & used:
                return None
            used |= interior
            paths[frozenset((a, b))] = p
        return paths

    def try_find_paths_backtrack(branches, pair_order):
        """Backtracking search: try different path choices for each pair."""
        bset = set(branches)
        used = set()
        paths = {}
        
        def backtrack(idx):
            if idx == len(pair_order):
                return True
            a, b = pair_order[idx]
            banned = used | bset
            banned.discard(a)
            banned.discard(b)
            
            # Find all simple paths up to a reasonable length limit
            all_paths = find_all_short_paths(a, b, banned, max_len=5)
            for p in all_paths:
                interior = set(p[1:-1])
                if interior & bset or interior & used:
                    continue
                used_copy = used | interior
                paths_copy = dict(paths)
                paths_copy[frozenset((a, b))] = p
                if backtrack(idx + 1):
                    used.update(interior)
                    paths.update(paths_copy)
                    return True
            return False
        
        if backtrack(0):
            return paths
        return None

    def find_all_short_paths(u, v, banned, max_len=5):
        """Find all simple paths from u to v with length <= max_len, avoiding banned."""
        if u == v:
            return []
        if v in adj[u]:
            return [[u, v]]
        
        results = []
        def dfs(cur, path, visited):
            if len(path) - 1 > max_len:
                return
            for nxt in sorted(adj[cur]):
                if nxt == v:
                    results.append(path + [v])
                    continue
                if nxt in visited or nxt in banned:
                    continue
                visited.add(nxt)
                dfs(nxt, path + [nxt], visited)
                visited.remove(nxt)
        
        visited = {u}
        dfs(u, [u], visited)
        return results

    def generate_pair_orders(branches):
        """Generate multiple pair orderings to try."""
        pairs = list(combinations(branches, 2))
        orders = []
        
        # Order 1: non-adjacent first, then by sum of degrees descending
        def key1(p):
            a, b = p
            if b in adj[a]:
                return (1, degrees[a] + degrees[b])
            else:
                return (0, degrees[a] + degrees[b])
        orders.append(sorted(pairs, key=key1))
        
        # Order 2: adjacent first (easier), then non-adjacent
        def key2(p):
            a, b = p
            if b in adj[a]:
                return (0, degrees[a] + degrees[b])
            else:
                return (1, degrees[a] + degrees[b])
        orders.append(sorted(pairs, key=key2))
        
        # Order 3: by sum of degrees descending
        orders.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]], reverse=True))
        
        # Order 4: by sum of degrees ascending
        orders.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        
        # Order 5: random
        import random as _r
        _rng = _r.Random(hash(tuple(sorted(branches))) % 10000)
        shuffled = list(pairs)
        _rng.shuffle(shuffled)
        orders.append(shuffled)
        
        # Order 6: reverse of order 1
        orders.append(list(reversed(orders[0])))
        
        # Order 7: by distance (use BFS distance as proxy - closer pairs first)
        def dist_key(p):
            a, b = p
            # Quick distance estimate using degree sum
            return degrees[a] + degrees[b]
        orders.append(sorted(pairs, key=dist_key))
        
        # Order 8: random with different seed
        _rng2 = _r.Random(hash(tuple(sorted(branches))) * 7 + 13)
        shuffled2 = list(pairs)
        _rng2.shuffle(shuffled2)
        orders.append(shuffled2)
        
        return orders

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

    def update_best(branches, paths):
        nonlocal best_count, best_result
        cnt = validate(branches, paths)
        if cnt > best_count:
            best_count = cnt
            best_result = (list(branches), paths)
            return True
        return False

    # Phase 1: Try candidate sets from high-degree vertices
    # Start with reasonable k values
    for k in range(6, 2, -1):
        if time_up():
            break
        candidates = []
        seen = set()
        
        # Top k by degree
        c = tuple(sorted(sorted_verts[:k]))
        candidates.append(sorted_verts[:k])
        seen.add(frozenset(c))
        
        # Variations: swap one vertex
        for i in range(min(k, len(sorted_verts))):
            if time_up():
                break
            for j in range(k, min(k + 8, n)):
                if time_up():
                    break
                c = list(sorted_verts[:k])
                c[i] = sorted_verts[j]
                c = sorted(c)
                fs = frozenset(c)
                if fs not in seen:
                    candidates.append(c)
                    seen.add(fs)
        
        # Random samples
        for _ in range(20):
            if time_up():
                break
            c = sorted(rng.sample(range(n), k))
            fs = frozenset(c)
            if fs not in seen:
                candidates.append(c)
                seen.add(fs)
        
        for cand in candidates:
            if time_up():
                break
            orders = generate_pair_orders(cand)
            for order in orders:
                if time_up():
                    break
                paths = try_find_paths(cand, order)
                if paths is not None:
                    if update_best(cand, paths):
                        break
            if best_count >= k:
                break
        if best_count >= k:
            break

    # Phase 2: Greedy growth with backtracking fallback
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        
        current_paths = None
        if current:
            orders = generate_pair_orders(current)
            for order in orders:
                current_paths = try_find_paths(current, order)
                if current_paths is not None:
                    break
        
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
                orders = generate_pair_orders(new_br)
                new_paths = None
                for order in orders:
                    new_paths = try_find_paths(new_br, order)
                    if new_paths is not None:
                        break
                if new_paths is None:
                    # Try backtracking approach
                    new_paths = try_find_paths_backtrack(new_br, orders[0])
                if new_paths is not None:
                    if update_best(new_br, new_paths):
                        current = new_br
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break

    # Phase 3: Try to grow from scratch with different seeds
    if not time_up() and best_count < n - 1:
        for start_k in [3, 4, 5]:
            if time_up():
                break
            seed_candidates = []
            seen_seeds = set()
            
            # Top k by degree
            c = sorted(sorted_verts[:start_k])
            seed_candidates.append(c)
            seen_seeds.add(frozenset(c))
            
            # Add some random seeds
            for _ in range(10):
                if time_up():
                    break
                c = sorted(rng.sample(range(n), start_k))
                fs = frozenset(c)
                if fs not in seen_seeds:
                    seed_candidates.append(c)
                    seen_seeds.add(fs)
            
            for seed in seed_candidates:
                if time_up():
                    break
                # Try to grow this seed
                current = list(seed)
                orders = generate_pair_orders(current)
                current_paths = None
                for order in orders:
                    current_paths = try_find_paths(current, order)
                    if current_paths is not None:
                        break
                if current_paths is None:
                    continue
                
                for _ in range(10):
                    if time_up():
                        break
                    cset = set(current)
                    cands = [v for v in range(n) if v not in cset]
                    cands.sort(key=lambda v: degrees[v], reverse=True)
                    added = False
                    for v in cands:
                        if time_up():
                            break
                        new_br = current + [v]
                        orders2 = generate_pair_orders(new_br)
                        new_paths = None
                        for order2 in orders2:
                            new_paths = try_find_paths(new_br, order2)
                            if new_paths is not None:
                                break
                        if new_paths is None:
                            new_paths = try_find_paths_backtrack(new_br, orders2[0])
                        if new_paths is not None:
                            if update_best(new_br, new_paths):
                                current = new_br
                                current_paths = new_paths
                                added = True
                                break
                    if not added:
                        break

    # Fallback
    if best_result is None:
        u, v = edges[0]
        best_result = ([u, v], {frozenset((u, v)): [u, v]})

    branches, paths = best_result
    return final_result(branches, paths)
