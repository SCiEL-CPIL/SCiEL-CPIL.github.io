"""Rewritten program: Maximum Independent Set on a candidate conflict graph."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def construct():
    # ---------- Build candidate pool ----------
    def gen_pool():
        vecs = []
        seen = set()

        def add(v):
            # normalize sign: make first nonzero entry positive
            t = list(v)
            for x in t:
                if x != 0:
                    if x < 0:
                        t = [-a for a in t]
                    break
            key = tuple(t)
            if key in seen:
                return
            if all(a == 0 for a in t):
                return
            seen.add(key)
            vecs.append(t)

        # norm^2 = 1
        for i in range(D):
            v = [0] * D
            v[i] = 1
            add(v)

        # norm^2 = 2
        for i in range(D):
            for j in range(i + 1, D):
                v = [0] * D
                v[i] = 1
                v[j] = 1
                add(v)

        # norm^2 = 3
        for i in range(D):
            for j in range(i + 1, D):
                for k in range(j + 1, D):
                    v = [0] * D
                    v[i] = v[j] = v[k] = 1
                    add(v)

        # norm^2 = 4 (type A: two 2's)
        for i in range(D):
            for j in range(i + 1, D):
                v = [0] * D
                v[i] = v[j] = 2
                add(v)

        # norm^2 = 4 (type B: four 1's)
        for i in range(D):
            for j in range(i + 1, D):
                for k in range(j + 1, D):
                    for m in range(k + 1, D):
                        v = [0] * D
                        v[i] = v[j] = v[k] = v[m] = 1
                        add(v)

        return vecs

    candidates = gen_pool()
    N = len(candidates)

    # ---------- Build conflict graph (adjacency as int64 bitsets) ----------
    # conflict if min(|u|,|v|)^2 - u.v < 2  (i.e. distance < max norm)
    norms2 = [sum(x * x for x in v) for v in candidates]
    adj = [0] * N
    for i in range(N):
        ni = norms2[i]
        vi = candidates[i]
        for j in range(i + 1, N):
            nj = norms2[j]
            mj = min(ni, nj)
            dot = 0
            vj = candidates[j]
            for t in range(D):
                dot += vi[t] * vj[t]
            if mj - dot < 2:
                # conflict: set bits
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # ---------- Deterministic branch-and-bound with Tomita coloring ----------
    best = [0]
    best_set = [0]  # bitmask of chosen vertices (may be large; use python int)

    # candidate order for greedy coloring (by degree descending)
    deg = [bin(adj[i]).count("1") for i in range(N)]
    order = sorted(range(N), key=lambda i: (-deg[i], i))
    pos = [0] * N
    for idx, v in enumerate(order):
        pos[v] = idx
    order_inv = list(range(N))  # placeholder

    def greedy_color(cand, conflict):
        # cand: list of vertices (already in order)
        # returns colors array aligned to cand, and list of color classes (sets as lists)
        colors = [0] * len(cand)
        color_classes = []  # list of lists
        # map vertex -> class index used for checking
        class_of = {}
        for idx, v in enumerate(cand):
            # find a color class that has no vertex conflicting with v
            # conflict among cand is determined by adj bitsets
            vbit = 1 << v
            placed = False
            # iterate classes; check conflict with each member
            for ci, members in enumerate(color_classes):
                ok = True
                for m in members:
                    # if m conflicts with v
                    if (adj[m] >> v) & 1:
                        ok = False
                        break
                if ok:
                    members.append(v)
                    colors[idx] = ci + 1
                    class_of[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[idx] = len(color_classes)
                class_of[v] = len(color_classes) - 1
        return colors, color_classes, class_of

    def branch_and_bound(cand, base):
        # cand: list of vertices (in `order` ordering), base: bitmask of chosen
        n = len(cand)
        if n == 0:
            return
        # quick prune: even taking all cannot beat best
        if len(base) + n <= best[0]:
            return
        colors, color_classes, class_of = greedy_color(cand, None)
        # upper bound = number of color classes
        if len(base) + len(color_classes) <= best[0]:
            return

        # expand in reverse order (from last vertex)
        # We'll iterate from the end of cand.
        # Maintain a temporary list of remaining candidates for recursion.
        # To reduce copies, we operate on lists.
        for idx in range(n - 1, -1, -1):
            v = cand[idx]
            # bound: base + (idx+1) is max we can get if we pick everything from 0..idx
            # but tighter: base + len(base)+... use color bound for the sub-list
            # Simple bound:
            if len(base) + (idx + 1) <= best[0]:
                continue
            # choose v
            new_base = base | (1 << v)
            # remaining candidates: those in cand[0:idx] that do NOT conflict with v
            vbit = 1 << v
            new_cand = []
            for u in cand[:idx]:
                if (adj[u] >> v) & 1:
                    continue
                new_cand.append(u)
            branch_and_bound(new_cand, new_base)
            # update best
            if bin(new_base).count("1") > best[0]:
                best[0] = bin(new_base).count("1")
                best_set[0] = new_base
            # do-not-choose v: continue loop (v is dropped from consideration)
            # also global prune
            if len(base) + (idx) <= best[0]:
                break

    # initial candidate set: all vertices, in degree order
    init_cand = order[:]
    branch_and_bound(init_cand, 0)

    # ---------- Materialize result ----------
    chosen_mask = best_set[0]
    result = []
    for i in range(N):
        if (chosen_mask >> i) & 1:
            result.append(candidates[i])

    if not result:
        # fallback
        return np.array([[1] + [0] * (D - 1)], dtype=np.int64)

    return np.array(result, dtype=np.int64)
# EVOLVE-BLOCK-END
