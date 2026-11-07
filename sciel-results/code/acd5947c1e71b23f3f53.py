"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

# Precompute the canonical edge set once (deterministic, not part of a restart).
_ALL_EDGES = frozenset((min(u, v), max(u, v)) for u, v in P['edges'])
_N = P['n']


def _cross_edges(L, R):
    s = set()
    for l in L:
        for r in R:
            s.add((min(l, r), max(l, r)))
    return s


def _expand_biclique(seed, remaining, adj, n, deadline, rng, all_side=True, max_iter=200):
    """Expand a biclique from a seed edge (u,v) into a (near) maximal biclique.

    Returns (L, R) as disjoint sets, or None if it cannot cover the seed edge.
    """
    u, v = seed
    if (min(u, v), max(u, v)) not in remaining:
        return None
    L = {u}
    R = {v}
    changed = True
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        if time.time() > deadline:
            break
        # Candidate vertices for each side (not already used on that side).
        cand_L = [x for x in range(n) if x not in L]
        cand_R = [y for y in range(n) if y not in R]
        if all_side:
            add_L = [x for x in cand_L if all((min(x, r), max(x, r)) in remaining for r in R)]
            add_R = [y for y in cand_R if all((min(l, y), max(l, y)) in remaining for l in L)]
        else:
            # Randomized single-vertex growth (explores different maximal bicliques).
            add_L = []
            for x in cand_L:
                if all((min(x, r), max(x, r)) in remaining for r in R):
                    add_L.append(x)
                    break
            add_R = []
            for y in cand_R:
                if all((min(l, y), max(l, y)) in remaining for l in L):
                    add_R.append(y)
                    break
        if add_L:
            if all_side:
                for x in add_L:
                    L.add(x)
            else:
                L.add(rng.choice(add_L))
            changed = True
        if add_R:
            if all_side:
                for y in add_R:
                    R.add(y)
            else:
                R.add(rng.choice(add_R))
            changed = True
    if L & R:
        return None
    if not L or not R:
        return None
    # Must still cover the seed edge.
    if (min(u, v), max(u, v)) not in _cross_edges(L, R):
        return None
    return (L, R)


def _greedy_partition(edges, adj, n, deadline, rng, seed_mode):
    """Run one greedy biclique set-cover pass. Returns list of (L,R) or None on timeout."""
    remaining = set(edges)
    bicliques = []
    while remaining:
        if time.time() > deadline:
            return None
        if seed_mode == 'edge':
            rem_list = list(remaining)
            if len(rem_list) > 50:
                step = len(rem_list) / 50.0
                idxs = sorted(set(int(i * step) for i in range(50)))
                idxs.append(0)
                idxs.append(len(rem_list) - 1)
                idxs = sorted(set(idxs))
                seeds = [rem_list[i] for i in idxs]
            else:
                seeds = rem_list
            best = None
            best_score = -1
            best_cov = set()
            for s in seeds:
                b = _expand_biclique(s, remaining, adj, n, deadline, rng, all_side=True)
                if b is None:
                    continue
                L, R = b
                cov = _cross_edges(L, R) & remaining
                if not cov:
                    continue
                score = len(cov) / max(1, len(L) + len(R))
                if score > best_score + 1e-12:
                    best_score = score
                    best = (L, R)
                    best_cov = cov
            if best is None:
                s = next(iter(remaining))
                best = ({s[0]}, {s[1]})
                best_cov = {s}
        else:  # 'vertex'
            deg = [0] * n
            for e in remaining:
                deg[e[0]] += 1
                deg[e[1]] += 1
            order = sorted(range(n), key=lambda x: (-deg[x], x))
            cands = [x for x in order[:12] if deg[x] > 0]
            if not cands:
                break
            best = None
            best_cov = set()
            best_score = -1
            for v in cands:
                R = set()
                for e in remaining:
                    if e[0] == v:
                        R.add(e[1])
                    elif e[1] == v:
                        R.add(e[0])
                if not R:
                    continue
                L = {v}
                changed = True
                it = 0
                while changed and it < 200:
                    changed = False
                    it += 1
                    if time.time() > deadline:
                        break
                    add_L = [x for x in range(n) if x not in L and all((min(x, r), max(x, r)) in remaining for r in R)]
                    add_R = [y for y in range(n) if y not in R and all((min(l, y), max(l, y)) in remaining for l in L)]
                    if add_L:
                        for x in add_L:
                            L.add(x)
                        changed = True
                    if add_R:
                        for y in add_R:
                            R.add(y)
                        changed = True
                if L & R:
                    continue
                cov = _cross_edges(L, R) & remaining
                if not cov:
                    continue
                score = len(cov) / max(1, len(L) + len(R))
                if score > best_score + 1e-12:
                    best_score = score
                    best = (L, R)
                    best_cov = cov
            if best is None:
                s = next(iter(remaining))
                best = ({s[0]}, {s[1]})
                best_cov = {s}
        bicliques.append(best)
        remaining -= best_cov
    return bicliques


def _refine(bicliques, adj, n, deadline):
    """Merge compatible bicliques and try vertex moves that enable merges."""
    host = _ALL_EDGES
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                L = L1 | L2
                R = R1 | R2
                if L & R:
                    continue
                ok = all(r in adj[l] for l in L for r in R)
                if not ok:
                    continue
                new_edges = _cross_edges(L, R)
                old_edges = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                if new_edges == old_edges:
                    bicliques[i] = (L, R)
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break

    if time.time() < deadline:
        for _ in range(6):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        new_L2 = L2 | {v}
                        if not new_L1 or not new_L2:
                            continue
                        if not all(r in adj[l] for l in new_L1 for r in R1):
                            continue
                        if not all(r in adj[l] for l in new_L2 for r in R2):
                            continue
                        merged_L = new_L1 | new_L2
                        merged_R = R1 | R2
                        if merged_L & merged_R:
                            continue
                        if not all(r in adj[l] for l in merged_L for r in merged_R):
                            continue
                        if _cross_edges(merged_L, merged_R) == (_cross_edges(L1, R1) | _cross_edges(L2, R2)):
                            bicliques[i] = (merged_L, merged_R)
                            bicliques.pop(j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    return bicliques


def _valid(bicliques, n):
    covered = set()
    for L, R in bicliques:
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for l in L:
            if not isinstance(l, int) or isinstance(l, bool):
                return False
            if l < 0 or l >= n:
                return False
        for r in R:
            if not isinstance(r, int) or isinstance(r, bool):
                return False
            if r < 0 or r >= n:
                return False
        for l in L:
            for r in R:
                k = (min(l, r), max(l, r))
                if k not in _ALL_EDGES:
                    return False
                if k in covered:
                    return False
                covered.add(k)
    if covered != _ALL_EDGES:
        return False
    return True


def _to_result(bicliques):
    out = []
    for L, R in bicliques:
        out.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
    return out


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin well under the 90s kill
    n = _N
    edges = _ALL_EDGES

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    rng = random.Random(20240517)

    # Deterministic baseline first (guarantees a valid fallback).
    best = None
    base = _greedy_partition(edges, adj, n, deadline, rng, 'edge')
    if base is not None:
        base = _refine(base, adj, n, deadline)
        if _valid(base, n):
            best = base

    # Randomized multi-restart exploration over different seed modes.
    modes = ['edge', 'vertex', 'edge', 'vertex', 'edge']
    mi = 0
    while time.time() < deadline - 6.0:
        mode = modes[mi % len(modes)]
        mi += 1
        cand = _greedy_partition(edges, adj, n, deadline, rng, mode)
        if cand is None:
            continue
        cand = _refine(cand, adj, n, deadline)
        if not _valid(cand, n):
            continue
        if best is None or len(cand) < len(best):
            best = cand
            if len(best) <= 3:
                break

    if best is None:
        # Absolute fallback: one biclique per edge (always valid).
        best = [({u}, {v}) for (u, v) in edges]

    return _to_result(best)
# EVOLVE-BLOCK-END
