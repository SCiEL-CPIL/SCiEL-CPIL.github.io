"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        C = list(range(k))
        I = list(range(k, n))
        m = len(I)
        # adjacency: for each i in I, which C-vertices are adjacent
        adj = [0] * m
        for idx, v in enumerate(I):
            mask = 0
            for u in range(k):
                if rng.random() < p:
                    mask |= (1 << u)
            adj[idx] = mask

        # C-edges as bitmasks
        Cedges = [(1 << u) | (1 << v) for u in range(k) for v in range(u + 1, k)]

        cliques = []
        used_i = [False] * m
        covered = 0  # bitmask of covered C-edges (list of masks instead for big k)
        # Represent covered C-edges as a set of (u,v)
        covered_edges = set()

        # Precompute pair cliques
        pairs = []
        for a in range(m):
            ma = adj[a]
            if ma == 0:
                continue
            for b in range(a + 1, m):
                mb = adj[b]
                if mb == 0:
                    continue
                inter = ma & mb
                # pair clique = {I[a], I[b]} ∪ inter
                # size = 2 + popcount(inter)
                pc = bin(inter).count('1')
                # The C-edges covered: all edges (u,v) where both u,v in inter
                # For scoring, count = pc*(pc-1)//2
                score = pc * (pc - 1) // 2
                if score >= 2:  # covers at least 2 C-edges (pc>=2)
                    pairs.append((score, a, b, inter, pc))
        pairs.sort(reverse=True)

        # Greedy: pick pairs that cover the most new C-edges
        for score, a, b, inter, pc in pairs:
            if used_i[a] or used_i[b]:
                continue
            if time.time() > deadline - 5:
                break
            # compute new C-edges covered
            new_count = 0
            verts = []
            # inter vertices
            for u in range(k):
                if inter & (1 << u):
                    verts.append(u)
            for x in range(len(verts)):
                for y in range(x + 1, len(verts)):
                    if (verts[x], verts[y]) not in covered_edges:
                        new_count += 1
            if new_count < 2:
                continue
            # accept
            used_i[a] = True
            used_i[b] = True
            clique = [I[a], I[b]] + verts
            cliques.append(clique)
            for x in range(len(verts)):
                for y in range(x + 1, len(verts)):
                    covered_edges.add((verts[x], verts[y]))

        # Now cover residual C-edges using single-I cliques
        # For each unused i, we can cover edges among N(i) in C
        # Greedy: for each residual edge (u,v), find an unused i adjacent to both,
        #   prefer i with largest neighborhood
        # Build residual edge list
        residual = [e for e in Cedges if e not in covered_edges]
        # For each i, the set of C-vertices it's adjacent to
        # We'll repeatedly: pick an unused i, cover all residual edges within N(i)
        # Actually better: for each residual edge, assign to an i
        # Simpler greedy: while residual edges remain, pick i (unused) that covers most residual edges
        # To keep it fast, do a few passes
        # Precompute for each i the list of C-vertices
        i_verts = []
        for idx in range(m):
            v = []
            mask = adj[idx]
            for u in range(k):
                if mask & (1 << u):
                    v.append(u)
            i_verts.append(v)

        # residual as set
        residual_set = set(residual)
        # Greedy set cover: each unused i covers residual edges within its N(i)
        # Represent i's coverable edges
        while residual_set:
            if time.time() > deadline - 3:
                break
            best_i = -1
            best_cover = 0
            best_covers = None
            for idx in range(m):
                if used_i[idx]:
                    continue
                verts = i_verts[idx]
                if len(verts) < 2:
                    continue
                # count residual edges within verts
                cnt = 0
                covers = []
                for x in range(len(verts)):
                    for y in range(x + 1, len(verts)):
                        e = (verts[x], verts[y])
                        if e in residual_set:
                            cnt += 1
                            covers.append(e)
                if cnt > best_cover:
                    best_cover = cnt
                    best_i = idx
                    best_covers = covers
            if best_i < 0 or best_cover == 0:
                # fallback: use pure C cliques (edges)
                for e in residual_set:
                    cliques.append([e[0], e[1]])
                residual_set.clear()
                break
            used_i[best_i] = True
            verts = i_verts[best_i]
            cliques.append([I[best_i]] + verts)
            for e in best_covers:
                residual_set.discard(e)
        # any leftover (shouldn't be)
        for e in residual_set:
            cliques.append([e[0], e[1]])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline - 10:
            # fallback: trivial
            results.append([list(e) for e in instance(*spec)])
            continue
        results.append(solve(n, k, p, seed))

    # Convert to lists of ints
    out = []
    for inst in results:
        lst = []
        for c in inst:
            lst.append([int(x) for x in c])
        out.append(lst)
    return out
# EVOLVE-BLOCK-END
