import time
import random as _rnd
import math

def construct():
    t_start = time.time()
    deadline = t_start + 115.0

    def generate_edges(n, p, seed):
        rng = _rnd.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        all_mask = (1 << n) - 1

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            PX = P | X
            if PX:
                best_p = -1
                best_cnt = -1
                pp = PX
                while pp:
                    lb = pp & (-pp)
                    p = lb.bit_length() - 1
                    cnt = (adj[p] & P).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_p = p
                    pp ^= lb
                candidates = P & ~(adj[best_p])
            else:
                candidates = P

            cp = candidates
            while cp:
                lb = cp & (-cp)
                u = lb.bit_length() - 1
                cp ^= lb
                bk(R | lb, P & adj[u], X & adj[u])
                P ^= lb
                X |= lb

        bk(0, all_mask, 0)
        return cliques

    def filter_cliques(cliques, min_size=2):
        return [c for c in cliques if len(c) >= min_size]

    def greedy_hitting_set(cliques, n):
        uncovered = set(range(len(cliques)))
        vertex_cliques = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(ci)

        hit = set()
        while uncovered:
            best_v = -1
            best_cnt = -1
            for v in range(n):
                cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_cnt <= 0:
                break
            hit.add(best_v)
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        return hit

    def max_disjoint_cliques_fast(cliques):
        if not cliques:
            return 0
        sorted_cliques = sorted(cliques, key=len)
        count = 0
        used = [False] * len(sorted_cliques)
        for i in range(len(sorted_cliques)):
            if used[i]:
                continue
            count += 1
            ci = sorted_cliques[i]
            for j in range(i + 1, len(sorted_cliques)):
                if not used[j] and not ci.isdisjoint(sorted_cliques[j]):
                    used[j] = True
        return count

    def branch_and_bound(cliques, n, time_limit):
        t0 = time.time()
        best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        
        num_cliques = len(cliques)
        vertex_clique_mask = [0] * n
        for ci, c in enumerate(cliques):
            mask_bit = 1 << ci
            for v in c:
                vertex_clique_mask[v] |= mask_bit
        
        use_bitmask = num_cliques < 60

        def solve_mask(remaining_mask, current_hit_size, current_hit):
            nonlocal best_hit, best_size, t0
            if time.time() - t0 > time_limit:
                return False
            if remaining_mask == 0:
                if current_hit_size < best_size:
                    best_size = current_hit_size
                    best_hit = set(current_hit)
                return True
            
            rem_indices = []
            m = remaining_mask
            while m:
                lb = m & (-m)
                ci = lb.bit_length() - 1
                rem_indices.append(ci)
                m ^= lb
            
            lb_val = max_disjoint_cliques_fast([cliques[i] for i in rem_indices])
            if current_hit_size + lb_val >= best_size:
                return True
            
            best_ci = -1
            best_clique_size = 999
            for ci in rem_indices:
                sz = len(cliques[ci])
                if sz < best_clique_size:
                    best_clique_size = sz
                    best_ci = ci
            
            if best_ci < 0:
                return True
            
            target_clique = cliques[best_ci]
            verts = list(target_clique)
            verts.sort(key=lambda v: (vertex_clique_mask[v] & remaining_mask).bit_count(), reverse=True)
            
            for v in verts:
                new_mask = remaining_mask & ~vertex_clique_mask[v]
                if not solve_mask(new_mask, current_hit_size + 1, current_hit | {v}):
                    return False
            return True

        def solve_set(remaining_set, current_hit_size, current_hit):
            nonlocal best_hit, best_size, t0
            if time.time() - t0 > time_limit:
                return False
            if not remaining_set:
                if current_hit_size < best_size:
                    best_size = current_hit_size
                    best_hit = set(current_hit)
                return True
            lb_val = max_disjoint_cliques_fast([cliques[i] for i in remaining_set])
            if current_hit_size + lb_val >= best_size:
                return True
            best_ci = min(remaining_set, key=lambda i: len(cliques[i]))
            target = cliques[best_ci]
            verts = list(target)
            verts.sort(key=lambda v: sum(1 for i in remaining_set if v in cliques[i]), reverse=True)
            for v in verts:
                new_set = set(i for i in remaining_set if v not in cliques[i])
                if not solve_set(new_set, current_hit_size + 1, current_hit | {v}):
                    return False
            return True

        if use_bitmask:
            all_mask = (1 << num_cliques) - 1
            solve_mask(all_mask, 0, set())
        else:
            solve_set(set(range(num_cliques)), 0, set())
        
        return best_hit

    def local_search(hit, cliques, n, time_limit):
        hit = set(hit)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            for v in list(hit):
                trial = hit - {v}
                valid = True
                for c in cliques:
                    if not any(v2 in trial for v2 in c):
                        valid = False
                        break
                if valid:
                    hit = trial
                    improved = True
                    break
            if not improved:
                for v in list(hit):
                    if time.time() > t0 + time_limit:
                        break
                    trial = hit - {v}
                    for w in range(n):
                        if w in hit:
                            continue
                        trial2 = trial | {w}
                        if len(trial2) < len(hit):
                            valid = True
                            for c in cliques:
                                if not any(v2 in trial2 for v2 in c):
                                    valid = False
                                    break
                            if valid:
                                hit = trial2
                                improved = True
                                break
                    if improved:
                        break
        return hit

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 10:
            results.append(list(range(n)))
            continue

        edges = generate_edges(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch(adj, n)
        cliques = filter_cliques(cliques, min_size=2)

        if not cliques:
            results.append([])
            continue

        hit = greedy_hitting_set(cliques, n)
        
        bnb_time = min(remaining * 0.6, 30.0)
        hit_bnb = branch_and_bound(cliques, n, bnb_time)
        
        if len(hit_bnb) < len(hit):
            hit = hit_bnb
            
        ls_time = remaining * 0.3
        hit = local_search(hit, cliques, n, ls_time)

        valid = True
        for c in cliques:
            if not any(v in hit for v in c):
                valid = False
                break
        
        if not valid:
            all_v = set()
            for c in cliques:
                all_v |= c
            hit = all_v

        results.append(sorted(hit))

    return results
