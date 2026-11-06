import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_one(n, p, seed, sub_deadline):
        edges = instance(n, p, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        all_edges = set(edges)
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd0 = {v for v in deg if deg[v] % 2 == 1}
        lower = max(1, len(odd0) // 2)

        def odd_set(rem):
            d = defaultdict(int)
            for u, v in rem:
                d[u] += 1
                d[v] += 1
            return {v for v in d if d[v] % 2 == 1}

        def dfs_path(u, v, rem, adj, rng):
            path = [u]
            visited = {u}
            cur = u
            while True:
                if cur == v:
                    return path
                nbrs = [x for x in adj[cur] if (min(cur, x), max(cur, x)) in rem and x not in visited]
                if not nbrs:
                    if len(path) <= 1:
                        return None
                    path.pop()
                    visited.discard(cur)
                    cur = path[-1]
                    continue
                # Warnsdorff: fewest onward options first, with random tie-break
                nbrs.sort(key=lambda x: (sum(1 for y in adj[x] if (min(x, y), max(x, y)) in rem and y not in visited), rng.random()))
                nxt = nbrs[0]
                path.append(nxt)
                visited.add(nxt)
                cur = nxt

        def find_path(u, v, rem, adj, rng, tries=2):
            best = None
            for _ in range(tries):
                cand = dfs_path(u, v, rem, adj, rng)
                if cand and (best is None or len(cand) > len(best)):
                    best = cand
            return best

        def free_path(s, rem, adj, rng, tries=2):
            best = None
            for _ in range(tries):
                path = [s]
                visited = {s}
                cur = s
                while True:
                    nbrs = [x for x in adj[cur] if (min(cur, x), max(cur, x)) in rem and x not in visited]
                    if not nbrs:
                        break
                    nbrs.sort(key=lambda x: (sum(1 for y in adj[x] if (min(x, y), max(x, y)) in rem and y not in visited), rng.random()))
                    nxt = nbrs[0]
                    path.append(nxt)
                    visited.add(nxt)
                    cur = nxt
                if len(path) > 1 and (best is None or len(path) > len(best)):
                    best = path
            return best

        def make_cover(rng):
            rem = set(all_edges)
            cur_odd = odd_set(rem)
            paths = []
            while rem:
                if time.time() > sub_deadline - 1.0:
                    for e in rem:
                        paths.append(list(e))
                    rem = set()
                    break
                if len(cur_odd) >= 2:
                    odd_list = list(cur_odd)
                    rng.shuffle(odd_list)
                    pairs = []
                    if len(odd_list) <= 12:
                        for i in range(len(odd_list)):
                            for j in range(i + 1, len(odd_list)):
                                pairs.append((odd_list[i], odd_list[j]))
                    else:
                        for _ in range(40):
                            i = rng.randrange(len(odd_list))
                            j = rng.randrange(len(odd_list))
                            if i != j:
                                pairs.append((odd_list[i], odd_list[j]))
                    best_path = None
                    for (u, v) in pairs:
                        cand = find_path(u, v, rem, adj, rng)
                        if cand and (best_path is None or len(cand) > len(best_path)):
                            best_path = cand
                    if best_path:
                        paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            a, b = best_path[i], best_path[i + 1]
                            rem.discard((min(a, b), max(a, b)))
                        cur_odd = odd_set(rem)
                    else:
                        e = next(iter(rem))
                        paths.append(list(e))
                        rem.discard(e)
                        cur_odd = odd_set(rem)
                else:
                    verts = list({x for e in rem for x in e})
                    rng.shuffle(verts)
                    best_path = None
                    for s in verts[:min(12, len(verts))]:
                        cand = free_path(s, rem, adj, rng)
                        if cand and (best_path is None or len(cand) > len(best_path)):
                            best_path = cand
                    if best_path:
                        paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            a, b = best_path[i], best_path[i + 1]
                            rem.discard((min(a, b), max(a, b)))
                        cur_odd = odd_set(rem)
                    else:
                        e = next(iter(rem))
                        paths.append(list(e))
                        rem.discard(e)
                        cur_odd = odd_set(rem)
            return paths

        best = None
        for k in range(8):
            if time.time() > sub_deadline - 1.5:
                break
            cov = make_cover(random.Random(777 + k))
            if best is None or len(cov) < len(best):
                best = cov
                if len(best) <= lower:
                    break
        if best is None:
            best = [list(e) for e in all_edges]
        return best

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    n_specs = len(specs)
    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 1.0:
            results.append([list(e) for e in instance(*spec)])
            continue
        # give more time to later (larger) specs
        frac = (i + 1) / n_specs
        sub_deadline = min(deadline - 0.5, time.time() + time_left * frac)
        results.append(solve_one(*spec, sub_deadline))
    return results
