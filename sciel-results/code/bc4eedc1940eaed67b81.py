"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    t_start = time.time()
    DEADLINE = 150.0
    MARGIN = 4.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed, deadline):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        edge_set = set(edges)
        cross = [(u, v) for (u, v) in edges if u < k and v >= k]
        I = list(range(k, n))
        m = len(I)

        # neighbors of each I-vertex inside the clique C
        nbr = [0] * m
        for idx, (u, v) in enumerate(cross):
            nbr[v - k] |= (1 << u)

        # all C-edges
        cedges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edge_index = {}
        for i, e in enumerate(cedges):
            edge_index[e] = i
        E = len(cedges)

        # a maximal clique inside C (all vertices)
        full = (1 << k) - 1

        # ---- initial feasible solution: each v gets a random maximal clique
        S = [0] * m
        for idx in range(m):
            base = nbr[idx]
            if base == 0:
                continue
            # random maximal extension: iterate vertices in random order, add if possible
            order = list(range(k))
            rng.shuffle(order)
            cur = base
            for u in order:
                if u >= k or (cur >> u) & 1:
                    continue
                mask = (1 << u)
                if (cur & mask) == 0 and (cur | mask) & ~base == 0:
                    # u is in base and u is a neighbor of every vertex in cur
                    if (cur & ~base) == 0:
                        cur |= mask
            S[idx] = cur

        # count how many times each C-edge is covered
        def edge_counts(S):
            cnt = [0] * E
            for idx in range(m):
                s = S[idx]
                if s == 0:
                    continue
                # edges inside s
                bits = s
                while bits:
                    lsb = bits & (-bits)
                    u = lsb.bit_length() - 1
                    bits ^= lsb
                    rest = s & (bits)  # remaining higher bits
                    while rest:
                        lsb2 = rest & (-rest)
                        v = lsb2.bit_length() - 1
                        rest ^= lsb2
                        cnt[edge_index[(u, v)]] += 1
            return cnt

        def objective(cnt):
            # minimize #over + #under, then minimize total cliques
            over = 0
            under = 0
            for c in cnt:
                if c > 1:
                    over += c - 1
                elif c == 0:
                    under += 1
            # total cliques = number of non-empty S
            nonempty = sum(1 for s in S if s != 0)
            return (over + under, nonempty)

        def popcount(x):
            return bin(x).count("1")

        # evaluate initial
        cnt = edge_counts(S)
        cur_obj = objective(cnt)

        best_S = list(S)
        best_obj = cur_obj

        # ---- simulated annealing
        T = 0.5
        T_min = 1e-4
        steps = 0
        while time.time() < deadline - MARGIN:
            steps += 1
            # pick a random I-vertex and a random move
            idx = rng.randrange(m)
            base = nbr[idx]
            cur = S[idx]
            r = rng.random()
            if r < 0.5:
                # add a random vertex to cur
                avail = base & ~cur
                if avail == 0:
                    continue
                lsb = avail & (-avail)
                new = cur | lsb
            elif r < 0.8:
                # remove a random vertex
                if cur == 0:
                    continue
                lsb = cur & (-cur)
                new = cur ^ lsb
            else:
                # swap: remove one, add one
                if cur == 0:
                    continue
                avail = base & ~cur
                if avail == 0:
                    continue
                lsb_rm = cur & (-cur)
                lsb_add = avail & (-avail)
                new = (cur ^ lsb_rm) | lsb_add
            if new == cur:
                continue
            # compute delta in objective quickly
            # recompute counts for this vertex only
            old_edges = []
            s = cur
            bits = s
            while bits:
                lsb = bits & (-bits)
                u = lsb.bit_length() - 1
                bits ^= lsb
                rest = s & bits
                while rest:
                    lsb2 = rest & (-rest)
                    v = lsb2.bit_length() - 1
                    rest ^= lsb2
                    old_edges.append(edge_index[(u, v)])
            new_edges = []
            s = new
            bits = s
            while bits:
                lsb = bits & (-bits)
                u = lsb.bit_length() - 1
                bits ^= lsb
                rest = s & bits
                while rest:
                    lsb2 = rest & (-rest)
                    v = lsb2.bit_length() - 1
                    rest ^= lsb2
                    new_edges.append(edge_index[(u, v)])
            # apply delta
            for e in old_edges:
                cnt[e] -= 1
            for e in new_edges:
                cnt[e] += 1
            new_obj = objective(cnt)
            S[idx] = new
            if new_obj <= cur_obj:
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_S = list(S)
            else:
                # accept with probability exp(-delta/T)
                import math
                delta = (new_obj[0] - cur_obj[0]) * 1.0 + (new_obj[1] - cur_obj[1]) * 0.05
                if rng.random() < math.exp(-delta / T):
                    cur_obj = new_obj
                else:
                    # revert
                    for e in new_edges:
                        cnt[e] -= 1
                    for e in old_edges:
                        cnt[e] += 1
                    S[idx] = cur
            # cool
            T = max(T_min, T * 0.9995)

        # rebuild final solution from best_S
        cnt = edge_counts(best_S)
        # verify feasibility
        for c in cnt:
            if c != 1:
                # fallback: use per-edge cliques (valid, just many)
                cliques = []
                for (u, v) in edges:
                    cliques.append([u, v])
                return cliques
        # build cliques
        cliques = []
        for idx in range(m):
            s = best_S[idx]
            if s == 0:
                continue
            v = I[idx]
            verts = [v]
            bits = s
            while bits:
                lsb = bits & (-bits)
                u = lsb.bit_length() - 1
                bits ^= lsb
                verts.append(u)
            cliques.append(verts)
        return cliques

    result = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        n, k, p, seed = spec
        # budget per instance
        remain = deadline - time.time() - MARGIN
        per = max(0.5, remain / max(1, len(specs)))
        cliques = solve(n, k, p, seed, min(deadline - MARGIN, time.time() + per))
        # ensure Python ints and lists
        cliques = [[int(x) for x in c] for c in cliques]
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
