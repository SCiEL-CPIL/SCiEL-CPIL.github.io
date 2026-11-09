import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 140

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def solve_instance(n, p, seed, time_budget):
        adj_mask_orig = get_graph(n, p, seed)
        if not any(adj_mask_orig):
            return []

        def iter_bits(x):
            while x:
                lsb = x & -x
                yield lsb.bit_length() - 1
                x ^= lsb

        def edge_common(adj, n):
            edges = []
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        edges.append(((adj[u] & adj[v]).bit_count(), u, v))
                    m ^= lsb
            return edges

        def pick_edge(edges, rng, mode):
            if not edges:
                return -1, -1
            if mode == 2:
                c, u, v = edges[rng.randint(0, len(edges) - 1)]
                return u, v
            if mode == 1:
                edges.sort(key=lambda x: (-x[0], x[1], x[2]))
            else:
                edges.sort(key=lambda x: (x[0], x[1], x[2]))
            if rng is None:
                return edges[0][1], edges[0][2]
            k = min(len(edges), max(8, len(edges) // 4))
            best_key = float('inf')
            best_u = best_v = -1
            for common, u, v in edges[:k]:
                key = common + rng.random()
                if key < best_key:
                    best_key = key
                    best_u, best_v = u, v
            return best_u, best_v

        def expand_clique_multi(seed_u, seed_v, adj, rng=None, top_k=3):
            """Try expanding from top-K candidates in common neighborhood, return largest clique."""
            U = adj[seed_u] & adj[seed_v]
            if not U:
                return [seed_u, seed_v]
            
            candidates = list(iter_bits(U))
            if not candidates:
                return [seed_u, seed_v]
            
            # Score candidates by degree in U
            scored = []
            for u in candidates:
                score = (adj[u] & U).bit_count()
                scored.append((score, u))
            scored.sort(key=lambda x: (-x[0], x[1]))
            
            if rng is not None and len(scored) > 1:
                # Add slight randomness to top candidates
                k = min(top_k, len(scored))
                top = scored[:k]
                if rng.random() < 0.3:
                    rng.shuffle(top)
            else:
                top = scored[:top_k]
            
            best_clique = [seed_u, seed_v]
            for _, start_w in top:
                C = [seed_u, seed_v, start_w]
                U_cur = adj[seed_u] & adj[seed_v] & adj[start_w]
                while U_cur:
                    best_u = -1
                    best_score = -1
                    best_score2 = -1
                    cands = list(iter_bits(U_cur))
                    if rng is not None and len(cands) > 1:
                        rng.shuffle(cands)
                    for u in cands:
                        score = (adj[u] & U_cur).bit_count()
                        score2 = adj[u].bit_count()
                        if score > best_score or (score == best_score and score2 > best_score2):
                            best_score = score
                            best_score2 = score2
                            best_u = u
                    if best_u == -1:
                        break
                    C.append(best_u)
                    U_cur &= adj[best_u]
                if len(C) > len(best_clique):
                    best_clique = C
            
            return best_clique

        def remove_clique_edges(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def is_clique(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def local_improve(cliques, adj, n, deadline):
            """Try to reduce clique count by absorbing small cliques into larger ones."""
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for a in range(len(cliques)):
                    if time.time() >= deadline:
                        break
                    Ca = cliques[a]
                    if len(Ca) < 2:
                        continue
                    for b in range(len(cliques)):
                        if a == b:
                            continue
                        Cb = cliques[b]
                        if len(Cb) < 2:
                            continue
                        # Try to move vertices from smaller to larger clique
                        if len(Ca) > len(Cb):
                            small, large = Cb, Ca
                            small_idx, large_idx = b, a
                        else:
                            small, large = Ca, Cb
                            small_idx, large_idx = a, b
                        
                        for w in list(small):
                            rest_small = [x for x in small if x != w]
                            new_large = large + [w]
                            if len(rest_small) >= 2 and is_clique(rest_small, adj) and is_clique(new_large, adj):
                                if len(small) == 2:
                                    # Remove the 2-clique
                                    cliques = [cliques[k] for k in range(len(cliques)) if k != small_idx]
                                    # Find the large clique in new list
                                    for i, c in enumerate(cliques):
                                        if c == large:
                                            cliques[i] = new_large
                                            break
                                    changed = True
                                    break
                            elif is_clique(new_large, adj):
                                # Move w to large even if small still valid
                                rest_small = [x for x in small if x != w]
                                if is_clique(rest_small, adj):
                                    cliques[small_idx] = rest_small
                                    for i, c in enumerate(cliques):
                                        if c == large:
                                            cliques[i] = new_large
                                            break
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
            return cliques

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None, mode=0):
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []
            while True:
                if time.time() > deadline:
                    break
                edges = edge_common(adj, n)
                u, v = pick_edge(edges, rng, mode)
                if u == -1:
                    break
                C = expand_clique_multi(u, v, adj, rng, top_k=3)
                if len(C) < 2:
                    C = [u, v]
                cliques.append(C)
                remove_clique_edges(C, adj)
            cliques = local_improve(cliques, adj, n, deadline)
            return cliques

        def validate(cliques, adj_mask_orig, n):
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                m = adj_mask_orig[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
                    m ^= lsb
            return True

        best_cliques = None
        best_count = float('inf')

        # Deterministic run first
        det_deadline = start_time + time_budget * 0.35
        det_cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline, mode=0)
        if validate(det_cliques, adj_mask_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)

        # Randomized restarts
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(1, int(remaining / 1.5))
            rng_master = random.Random(seed + 1000)
            modes = [0, 0, 0, 1, 2]
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1:
                    break
                rng = random.Random(rng_master.randint(0, 2 ** 31))
                mode = modes[r % len(modes)]
                restart_deadline = min(start_time + time_budget - 1,
                                       time.time() + max(0.5, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng,
                                               deadline=restart_deadline, mode=mode)
                    if validate(cliques, adj_mask_orig, n):
                        if len(cliques) < best_count:
                            best_count = len(cliques)
                            best_cliques = cliques
                except Exception:
                    continue

        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                m = adj_mask_orig[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        best_cliques.append([u, v])
                    m ^= lsb

        return best_cliques

    results = []
    t1 = min(55, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))

    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results
