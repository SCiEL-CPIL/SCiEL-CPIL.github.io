"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = 132.0  # safety margin under the 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _time_ok():
        return (time.time() - t0) < DEADLINE

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        E = set((min(u, v), max(u, v)) for (u, v) in edges)
        if not E:
            return []

        # candidate cliques: {s}+N_I(s) for s in C, and {u,v} for u,v in C
        candidates = []
        for s in range(k):
            nb = sorted(v for (a, b) in E for v in [b if a == s else (a if b == s else -1)] if v >= k)
            if nb:
                candidates.append(tuple([s] + nb))
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in E:
                    candidates.append((u, v))
        if not candidates:
            # fallback: every edge as a clique (valid)
            return [list(e) for e in E]

        cset = [set(c) for c in candidates]
        cedges = []
        for c in cset:
            ce = set()
            L = list(c)
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    a, b = L[i], L[j]
                    ce.add((a, b) if a < b else (b, a))
            cedges.append(ce)

        best = None
        best_count = None
        # greedy seed
        covered = set()
        active = list(range(len(candidates)))
        order = sorted(active, key=lambda i: -len(cedges[i] - covered))
        chosen = []
        for i in order:
            if not _time_ok():
                break
            g = cedges[i] - covered
            if g:
                chosen.append(i)
                covered |= cedges[i]
                if covered == E:
                    break
        if covered == E:
            best = list(chosen)
            best_count = len(chosen)

        # local search
        it = 0
        while best is not None and _time_ok():
            it += 1
            if it % 200 == 0 and not _time_ok():
                break
            improved = False
            for idx in range(len(best)):
                if not _time_ok():
                    break
                cand = best[idx]
                freed = cedges[cand] - (covered - cedges[cand])
                if not freed:
                    # removing cand keeps cover; drop it
                    best.pop(idx)
                    covered -= cedges[cand]
                    improved = True
                    break
                # try to re-cover freed edges with a single other clique
                # find a clique (not in best) covering all freed
                found = None
                for j in range(len(candidates)):
                    if j in best:
                        continue
                    if freed <= cedges[j]:
                        found = j
                        break
                if found is not None:
                    # swap: remove cand, add found
                    newcount = len(best)
                    best[idx] = found
                    covered = (covered - cedges[cand]) | cedges[found]
                    improved = True
                    break
            if not improved:
                break
            if len(best) < best_count:
                best_count = len(best)

        if best is None:
            # fallback valid: each edge
            return [list(e) for e in E]
        return [list(cset[i]) for i in best]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if not _time_ok():
            out.append([list(e) for e in instance(*spec)])
            continue
        res = solve(*spec)
        # validate & sanitize
        E = set((min(u, v), max(u, v)) for (u, v) in instance(*spec))
        seen = set()
        ok = True
        for c in res:
            L = list(c)
            if len(L) < 2 or len(set(L)) != len(L):
                ok = False
                break
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    a, b = L[i], L[j]
                    e = (a, b) if a < b else (b, a)
                    if e not in E or e in seen:
                        ok = False
                        break
                    seen.add(e)
                if not ok:
                    break
            if not ok:
                break
        if not ok or seen != E:
            res = [list(e) for e in E]
        out.append(res)
    return out
# EVOLVE-BLOCK-END
