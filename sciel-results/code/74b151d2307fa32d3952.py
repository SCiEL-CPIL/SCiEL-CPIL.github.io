"""Rewritten program: split-structure-aware clique cover via branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import math
import random
from random import Random
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 120.0
_MARGIN = 10.0


def _time_ok():
    return (time.time() - _DEADLINE) < -_MARGIN


def _is_clique_set(cliques, adj, n):
    # every clique must be a true clique
    for c in cliques:
        if len(c) < 2:
            return False
        for i in range(len(c)):
            ci = c[i]
            for j in range(i + 1, len(c)):
                cj = c[j]
                if cj not in adj[ci]:
                    return False
    # every edge must be covered exactly once
    edge_count = 0
    seen = set()
    for c in cliques:
        L = len(c)
        for i in range(L):
            ci = c[i]
            for j in range(i + 1, L):
                cj = c[j]
                key = (ci, cj) if ci < cj else (cj, ci)
                if key in seen:
                    return False
                seen.add(key)
                edge_count += 1
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if (u, v) not in seen:
                    return False
    return edge_count == len(seen)


def _cover_from_seed(n, k, p, seed):
    # The seed generator / trivial cover (one clique per edge).
    rng = Random(seed)
    edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
    cross = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
    all_edges = edges + cross
    return [[e[0], e[1]] for e in all_edges]


def _try_improve(cliques, adj, n, k, time_budget):
    """Greedy: merge any two cliques whose union is a clique (reduces count)."""
    cliques = [list(c) for c in cliques]
    deadline = time.time() + time_budget
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # For each pair, check if union is a clique.
        m = len(cliques)
        for a in range(m):
            if time.time() > deadline:
                break
            A = cliques[a]
            sA = set(A)
            for b in range(a + 1, m):
                B = cliques[b]
                # quick check: every vertex of B must be adjacent to every vertex of A
                ok = True
                for v in B:
                    if v in sA:
                        continue
                    for u in A:
                        if v not in adj[u]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    merged = list(A)
                    for v in B:
                        if v not in sA:
                            merged.append(v)
                    cliques[a] = merged
                    cliques[b] = []
                    changed = True
        cliques = [c for c in cliques if c]
    return cliques


def _bb_clique_cover(edges, adj, n, k, time_budget):
    """Branch-and-bound over candidate cliques to cover all edges exactly once.

    Candidate cliques: all maximal cliques of the graph.
    """
    deadline = time.time() + time_budget
    E = [(u, v) for (u, v) in edges if u < v]
    if not E:
        return []
    edge_id = {e: i for i, e in enumerate(E)}
    m = len(E)
    # Build candidate cliques (maximal).
    # Use Bron–Kerbosch with pivot on the full graph.
    candidates = []

    def bronk(R, P, X):
        if not _time_ok():
            return
        if not P and not X:
            if len(R) >= 2:
                candidates.append(list(R))
            return
        # pivot
        U = P | X
        if U:
            # pick pivot with max degree in P
            best = None
            best_d = -1
            for u in U:
                d = 0
                for v in P:
                    if v in adj[u]:
                        d += 1
                if d > best_d:
                    best_d = d
                    best = u
            P_minus = [v for v in P if v not in adj[best]]
        else:
            P_minus = list(P)
        for v in P_minus:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
            if not P and not X:
                break

    bronk(set(), set(range(n)), set())
    if not _time_ok():
        return None

    # Deduplicate candidates and sort by size desc.
    cand_set = set()
    for c in candidates:
        cand_set.add(tuple(sorted(c)))
    candidates = sorted(cand_set, key=lambda c: (-len(c), c))
    # Precompute edge-mask for each candidate (as a Python int bitmask if m <= 64 else set)
    if m <= 64:
        cand_masks = []
        for c in candidates:
            mask = 0
            L = len(c)
            for i in range(L):
                ci = c[i]
                for j in range(i + 1, L):
                    cj = c[j]
                    mask |= (1 << edge_id[(ci, cj) if ci < cj else (cj, ci)])
            cand_masks.append(mask)
        full = (1 << m) - 1
    else:
        cand_masks = []
        for c in candidates:
            s = set()
            L = len(c)
            for i in range(L):
                ci = c[i]
                for j in range(i + 1, L):
                    cj = c[j]
                    s.add(edge_id[(ci, cj) if ci < cj else (cj, ci)])
            cand_masks.append(s)
        full = set(range(m))

    # Upper bound: greedy set cover (approx) for initial best.
    best = [None]
    best_count = [m + 1]

    if m <= 64:
        # Greedy set cover for initial UB
        remaining = full
        chosen = []
        while remaining:
            best_c = None
            best_gain = 0
            for idx, mask in enumerate(cand_masks):
                gain = bin(mask & remaining).count("1")
                if gain > best_gain:
                    best_gain = gain
                    best_c = idx
            if best_c is None:
                break
            chosen.append(best_c)
            remaining &= ~cand_masks[best_c]
        if not remaining:
            best_count[0] = len(chosen)
            best[0] = list(chosen)
    else:
        remaining = set(full)
        chosen = []
        while remaining:
            best_c = None
            best_gain = 0
            for idx, s in enumerate(cand_masks):
                gain = len(s & remaining)
                if gain > best_gain:
                    best_gain = gain
                    best_c = idx
            if best_c is None:
                break
            chosen.append(best_c)
            remaining -= cand_masks[best_c]
        if not remaining:
            best_count[0] = len(chosen)
            best[0] = list(chosen)

    # Lower bound: max over edges of 1 (trivial) -> use greedy packing lower bound
    # Simple LB: for each edge, at least 1 clique -> LB = m (too weak). Use:
    # each clique covers at most max_clique_edges edges.
    max_cover = max((bin(mm).count("1") if m <= 64 else len(mm) for mm in cand_masks), default=1)
    lb_base = (m + max_cover - 1) // max_cover

    if m <= 64:
        chosen = []
        covered = 0

        def rec(remaining, depth):
            nonlocal covered
            if not _time_ok():
                return
            if not remaining:
                if len(chosen) < best_count[0]:
                    best_count[0] = len(chosen)
                    best[0] = list(chosen)
                return
            # LB
            if depth + lb_base > best_count[0]:
                return
            # pick an uncovered edge (first set bit)
            lsb = remaining & (-remaining)
            bit = (lsb.bit_length() - 1)
            # candidates covering this edge
            for idx, mask in enumerate(cand_masks):
                if (mask & lsb) and (mask & remaining) == mask:
                    # mask must be subset of remaining (exact cover constraint)
                    chosen.append(idx)
                    rec(remaining & ~mask, depth + 1)
                    chosen.pop()
                    if not _time_ok():
                        return
        rec(full, 0)
    else:
        chosen = []

        def rec(remaining, depth):
            if not _time_ok():
                return
            if not remaining:
                if len(chosen) < best_count[0]:
                    best_count[0] = len(chosen)
                    best[0] = list(chosen)
                return
            if depth + lb_base > best_count[0]:
                return
            # pick a specific uncovered edge: choose the one with fewest candidate supersets
            # (heuristic) - pick first for simplicity
            target = next(iter(remaining))
            for idx, s in enumerate(cand_masks):
                if target in s and s <= remaining:
                    chosen.append(idx)
                    rec(remaining - s, depth + 1)
                    chosen.pop()
                    if not _time_ok():
                        return
        rec(full, 0)

    if best[0] is None:
        return None
    return [list(candidates[i]) for i in best[0]]


def _build_adj(n, k, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(k):
        for v in range(u + 1, k):
            adj[u].add(v)
            adj[v].add(u)
    for v in range(k, n):
        for u in range(k):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _solve_one(n, k, p, seed, time_budget):
    adj = _build_adj(n, k, p, seed)
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    seed_cliques = _cover_from_seed(n, k, p, seed)
    # Improve via greedy merge
    merged = _try_improve(seed_cliques, adj, n, k, time_budget * 0.4)
    if not _time_ok():
        return merged
    # Try branch-and-bound
    bb = _bb_clique_cover(edges, adj, n, k, time_budget * 0.5)
    if bb is not None and _is_clique_set(bb, adj, n):
        if len(bb) < len(merged):
            return bb
    return merged


def construct():
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    per = 35.0
    results = []
    for spec in specs:
        if not _time_ok():
            # fallback to seed cover
            n, k, p, seed = spec
            results.append(_cover_from_seed(n, k, p, seed))
            continue
        cliques = _solve_one(spec[0], spec[1], spec[2], spec[3], per)
        # ensure types are Python ints and lists
        cliques = [[int(v) for v in c] for c in cliques]
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
