import time
import math
import random as _random
import sys

def construct():
    sys.setrecursionlimit(100000)
    start_time = time.time()
    DEADLINE = 140.0  # explicit wall-clock safety margin

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj_bitmask(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        """All maximal cliques (size >= 2) as lists of vertex indices (with pivoting)."""
        cliques = []
        full_mask = (1 << n) - 1

        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                if R_mask & (R_mask - 1):  # size >= 2
                    verts = [i for i in range(n) if (R_mask >> i) & 1]
                    cliques.append(verts)
                return
            P_union_X = P_mask | X_mask
            if P_union_X == 0:
                return
            # pivot maximising |P & N(u)|
            max_inter = -1
            pivot = -1
            temp = P_union_X
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                cnt = (P_mask & adj[u]).bit_count()
                if cnt > max_inter:
                    max_inter = cnt
                    pivot = u
                temp ^= lsb
            if pivot >= 0:
                candidates = P_mask & ~adj[pivot]
            else:
                candidates = P_mask
            if candidates == 0:
                return
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R_mask | lsb, P_mask & adj[v], X_mask & adj[v])
                P_mask &= ~lsb
                X_mask |= lsb
                temp ^= lsb

        bk(0, full_mask, 0)
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        adj = build_adj_bitmask(edges, n)
        cliques = bron_kerbosch(adj, n)

        # ---- reduce duplicate / superset maximal cliques ----
        uniq = {}
        for c in cliques:
            uniq[frozenset(c)] = True
        clique_sets = list(uniq.keys())
        filtered = []
        for i, s in enumerate(clique_sets):
            contained = False
            for j, t in enumerate(clique_sets):
                if i != j and s < t:
                    contained = True
                    break
            if not contained:
                filtered.append(s)
        clique_sets = filtered
        m = len(clique_sets)
        if m == 0:
            return []

        # vertex -> bitmask of cliques it hits
        vc = [0] * n
        for ci, s in enumerate(clique_sets):
            bit = 1 << ci
            for v in s:
                vc[v] |= bit
        full_cover = (1 << m) - 1

        # ---- candidate pruning: keep non-dominated vertices (coverage sets) ----
        cands = [v for v in range(n) if vc[v] != 0]
        cands.sort(key=lambda v: (-vc[v].bit_count(), v))
        keep = []
        for v in cands:
            dom = False
            for u in keep:
                if vc[v] & ~vc[u] == 0:  # C_v subset C_u  -> v dominated
                    dom = True
                    break
            if not dom:
                keep.append(v)
        cands = keep
        K = len(cands)

        # candidate -> index
        cidx = {v: i for i, v in enumerate(cands)}

        # per-clique list of candidate indices
        cliq_cands = []
        for s in clique_sets:
            li = []
            for v in s:
                if v in cidx:
                    li.append(cidx[v])
            cliq_cands.append(li)

        # candidate coverage as frozenset of clique indices (for matching)
        cand_clq = []
        for i, v in enumerate(cands):
            mm = vc[v]
            ss = set()
            t = mm
            while t:
                lsb = t & (-t)
                ss.add(lsb.bit_length() - 1)
                t ^= lsb
            cand_clq.append(ss)

        def coverage(i, unc):
            return (vc[cands[i]] & unc).bit_count()

        # ---- greedy hitting set ----
        def greedy(rng=None):
            selected = []
            sel_set = set()
            unc = full_cover
            while unc:
                best_i = -1
                best_cov = 0
                for i in range(K):
                    if i in sel_set:
                        continue
                    c = coverage(i, unc)
                    if c > best_cov or (c == best_cov and c > 0 and best_i >= 0 and
                                        (rng is not None and rng.random() < 0.3 and c == best_cov) or
                                        (c == best_cov and c > 0 and best_i < 0)):
                        # tie-break: prefer larger coverage, else random
                        if c > best_cov:
                            best_cov, best_i = c, i
                        elif c == best_cov and c > 0:
                            if rng is not None and rng.random() < 0.5:
                                best_i = i
                    elif c > best_cov:
                        best_cov, best_i = c, i
                if best_i < 0:
                    break
                selected.append(best_i)
                sel_set.add(best_i)
                unc &= ~vc[cands[best_i]]
            return selected

        init = greedy()
        best_size = len(init)
        best = set(init)
        for r in range(4):
            if time.time() - start_time > DEADLINE:
                break
            s = greedy(_random.Random(seed * 97 + r * 131))
            if len(s) < best_size:
                best_size, best = len(s), set(s)

        # ---- maximum matching on candidate conflict graph (lower bound) ----
        def matching_lb(unc):
            # build active cliques
            active = []
            t = unc
            while t:
                lsb = t & (-t)
                active.append(lsb.bit_length() - 1)
                t ^= lsb
            if not active:
                return 0
            if len(active) == 1:
                return 1
            # conflict graph: two candidates are adjacent if they share NO active clique
            # we want a matching of candidates that must be chosen distinct => lower bound
            # greedy matching on conflict graph of candidates restricted to active coverage
            cand_active = []
            for i in range(K):
                inter = vc[cands[i]] & unc
                if inter:
                    cand_active.append(i)
            if not cand_active:
                return 0
            # conflict adjacency: candidates i,j conflict if (vc_i & vc_j & unc) == 0
            matched = {}
            cnt = 0
            order = sorted(cand_active, key=lambda i: -vc[cands[i]].bit_count())
            for i in order:
                if i in matched:
                    continue
                found = False
                for j in cand_active:
                    if j in matched or j == i:
                        continue
                    if (vc[cands[i]] & vc[cands[j]] & unc) == 0:
                        matched[i] = j
                        matched[j] = i
                        cnt += 1
                        found = True
                        break
                # if no conflict partner, i is still a required distinct pick
            # each matched pair = 2 candidates for 2 disjoint cliques sets;
            # unmatched candidates = 1 each
            return cnt * 2 - cnt + (len(cand_active) - 2 * cnt) if False else \
                   (2 * cnt + (len(cand_active) - 2 * cnt)) // 1 if False else \
                   (cnt + (len(cand_active) - 2 * cnt))

        # ---- branch and bound ----
        def branch(selected, unc, depth):
            nonlocal best_size, best
            if unc == 0:
                if len(selected) < best_size:
                    best_size, best = len(selected), set(selected)
                return
            if len(selected) >= best_size:
                return
            if time.time() - start_time > DEADLINE:
                return
            # LB1: max single coverage
            maxc = 0
            for i in range(K):
                if i in selected:
                    continue
                c = coverage(i, unc)
                if c > maxc:
                    maxc = c
            if maxc == 0:
                return
            lb1 = len(selected) + math.ceil(unc.bit_count() / maxc)
            lb2 = len(selected) + matching_lb(unc)
            if max(lb1, lb2) >= best_size:
                return
            # choose uncovered clique with fewest available candidates
            tgt = -1
            minav = 10 ** 9
            t = unc
            while t:
                lsb = t & (-t)
                ci = lsb.bit_length() - 1
                av = 0
                for i in cliq_cands[ci]:
                    if i not in selected:
                        av += 1
                if av < minav:
                    minav, tgt = av, ci
                    if av == 1:
                        break
                t ^= lsb
            if tgt < 0:
                return
            opts = [i for i in cliq_cands[tgt] if i not in selected]
            opts.sort(key=lambda i: -coverage(i, unc))
            for i in opts:
                branch(selected | {i}, unc & ~vc[cands[i]], depth + 1)

        branch(set(), full_cover, 0)

        # ---- iterative-repair / min-sum vertex-cover style polish ----
        def repair_polish(sol):
            nonlocal best_size, best
            sol = set(sol)
            # phase 1: remove redundant vertices
            changed = True
            while changed:
                changed = False
                for i in list(sol):
                    cov = 0
                    for j in sol:
                        if j != i:
                            cov |= vc[cands[j]]
                    if cov == full_cover:
                        sol.discard(i)
                        changed = True
                        break
            if len(sol) < best_size:
                best_size, best = len(sol), set(sol)
            # phase 2: min-sum swaps (replace a high-load vertex with a low-load one)
            def load(i):
                return 0
            improved = True
            while improved:
                improved = False
                # current per-clique counts
                counts = [0] * m
                for j in sol:
                    t = vc[cands[j]]
                    while t:
                        lsb = t & (-t)
                        counts[lsb.bit_length() - 1] += 1
                        t ^= lsb
                # try each (remove i, add k) that keeps all counts >= 1
                for i in list(sol):
                    vc_i = vc[cands[i]]
                    # cliques uniquely covered by i (count==1 and i in vc_i)
                    crit = []
                    t = vc_i
                    while t:
                        lsb = t & (-t)
                        ci = lsb.bit_length() - 1
                        if counts[ci] == 1:
                            crit.append(ci)
                        t ^= lsb
                    for k in range(K):
                        if k in sol or k == i:
                            continue
                        ok = True
                        vc_k = vc[cands[k]]
                        for ci in crit:
                            if not (vc_k >> ci) & 1:
                                ok = False
                                break
                        if not ok:
                            continue
                        # evaluate net change in max load
                        # compute new counts
                        newc = counts[:]
                        t = vc_i
                        while t:
                            lsb = t & (-t)
                            newc[lsb.bit_length() - 1] -= 1
                            t ^= lsb
                        t = vc_k
                        while t:
                            lsb = t & (-t)
                            newc[lsb.bit_length() - 1] += 1
                            t ^= lsb
                        if min(newc) >= 1 and max(newc) < max(counts):
                            sol.discard(i)
                            sol.add(k)
                            improved = True
                            if len(sol) < best_size:
                                best_size, best = len(sol), set(sol)
                            break
                    if improved:
                        break
            return sol

        if best:
            repair_polish(best)

        out = sorted(cands[i] for i in best)
        if len(out) < 2:
            # safety: ensure a valid (non-trivial) transversal
            out = sorted(cands[i] for i in init)
        return out

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        results.append(solve_instance(n, p, seed))
    return results
