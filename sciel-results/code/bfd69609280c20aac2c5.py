"""Rewritten program for clique partition of split graphs."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # N[idx] = bitmask of C-vertices adjacent to I-vertex (idx + k)
        N = [0] * (n - k)
        for u, v in edges:
            if u < k and v >= k:
                N[v - k] |= (1 << u)
            elif v < k and u >= k:
                N[u - k] |= (1 << v)

        # rem[u] = bitmask of vertices v where C-edge (u,v) is not yet covered
        # Initially all pairs in C are uncovered
        rem = [0] * k
        for u in range(k):
            rem[u] = ((1 << k) - 1) ^ (1 << u)

        cliques = []

        # Phase 1: Process I-vertices by decreasing neighborhood size
        I_order = sorted(range(n - k), key=lambda idx: bin(N[idx]).count('1'), reverse=True)

        for idx in I_order:
            if time.time() > deadline:
                break
            i = idx + k
            mask = N[idx]
            if mask == 0:
                continue

            while mask:
                if time.time() > deadline:
                    break

                # Greedy max clique in subgraph induced by mask (using uncovered C-edges)
                S = []
                candidates = mask
                while candidates:
                    if time.time() > deadline:
                        break
                    best_v = -1
                    best_deg = -1
                    cm = candidates
                    while cm:
                        lsb = cm & (-cm)
                        v = lsb.bit_length() - 1
                        deg = bin(rem[v] & candidates).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                        cm ^= lsb
                    if best_v < 0:
                        break
                    S.append(best_v)
                    candidates &= rem[best_v]
                    candidates &= ~(1 << best_v)

                if not S:
                    lsb = mask & (-mask)
                    S = [lsb.bit_length() - 1]

                clique = sorted([i] + S)
                cliques.append(clique)

                # Mark edges within S as covered
                for a_idx in range(len(S)):
                    for b_idx in range(a_idx + 1, len(S)):
                        a, b = S[a_idx], S[b_idx]
                        rem[a] &= ~(1 << b)
                        rem[b] &= ~(1 << a)

                for v in S:
                    mask &= ~(1 << v)

        # Phase 2: Cover remaining C-edges with C-only cliques
        active = set()
        for u in range(k):
            if rem[u]:
                active.add(u)

        while active:
            if time.time() > deadline:
                break

            # Pick vertex with max degree in active subgraph
            best_v = -1
            best_deg = -1
            for v in active:
                cnt = 0
                for a in active:
                    if a != v and (rem[v] >> a) & 1:
                        cnt += 1
                if cnt > best_deg:
                    best_deg = cnt
                    best_v = v

            if best_v < 0 or best_deg == 0:
                break

            # Greedy clique expansion
            S = [best_v]
            candidates = rem[best_v]
            active_mask = 0
            for a in active:
                active_mask |= (1 << a)
            candidates &= active_mask
            candidates &= ~(1 << best_v)

            while candidates:
                if time.time() > deadline:
                    break
                best_w = -1
                best_wdeg = -1
                cm = candidates
                while cm:
                    lsb = cm & (-cm)
                    w = lsb.bit_length() - 1
                    deg = bin(rem[w] & candidates).count('1')
                    if deg > best_wdeg:
                        best_wdeg = deg
                        best_w = w
                    cm ^= lsb
                if best_w < 0:
                    break
                S.append(best_w)
                candidates &= rem[best_w]
                candidates &= ~(1 << best_w)

            if len(S) < 2:
                nbrs = [a for a in active if a != best_v and (rem[best_v] >> a) & 1]
                if nbrs:
                    S = [best_v, nbrs[0]]
                else:
                    active.discard(best_v)
                    continue

            S_list = sorted(S)
            cliques.append(S_list)

            for a_idx in range(len(S_list)):
                for b_idx in range(a_idx + 1, len(S_list)):
                    a, b = S_list[a_idx], S_list[b_idx]
                    rem[a] &= ~(1 << b)
                    rem[b] &= ~(1 << a)

            active = set(v for v in active if rem[v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
