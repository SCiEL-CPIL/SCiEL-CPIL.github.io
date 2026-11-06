"""Strong edge colouring: build the conflict graph, then use a fast simulated-annealing
style local search that (a) tries to recolour individual edges to free a colour class,
and (b) merges colour classes when compatible.  We run several restarts with different
random orderings and keep the best solution.  A hard deadline guard ensures we return
in time.
"""
import time
from random import Random
import sys


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        conf = [set() for _ in range(m)]
        v_to_e = {}
        for i, (u, v) in enumerate(edges):
            v_to_e.setdefault(u, []).append(i)
            v_to_e.setdefault(v, []).append(i)
        for i in range(m):
            u, v = edges[i]
            near = {u, v} | (adj.get(u, set()) | adj.get(v, set()))
            for x in near:
                for j in v_to_e.get(x, []):
                    if j != i and j > i:
                        conf[i].add(j)
                        conf[j].add(i)
        return conf

    def greedy(col, order, conf):
        col = [-1] * len(order)
        for i in order:
            taken = set()
            for j in conf[i]:
                c = col[j]
                if c >= 0:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            col[i] = c
        return col

    def k_of(col):
        return max(col) + 1 if col else 0

    def build_classes(col):
        cls = {}
        for i, c in enumerate(col):
            cls.setdefault(c, set()).add(i)
        return cls

    def reindex(col):
        present = sorted(set(col))
        if not present:
            return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def try_merge(col, cls, c1, c2, conf):
        """Try to merge colour class c2 into c1. Returns True if successful."""
        s1 = cls.get(c1)
        s2 = cls.get(c2)
        if s1 is None or s2 is None:
            return False
        for e in s2:
            if conf[e] & s1:
                return False
        for e in s2:
            col[e] = c1
        s1 |= s2
        del cls[c2]
        return True

    def local_improve(col, conf, m, deadline, rng):
        col = list(col)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 500:
            improved = False
            guard += 1
            cls = build_classes(col)
            
            # --- merge pass: fold one whole class into another when compatible ---
            merged = True
            while merged and time.time() < deadline:
                merged = False
                cols = sorted(cls.keys())
                for idx, c1 in enumerate(cols):
                    if c1 not in cls:
                        continue
                    s1 = cls[c1]
                    for c2 in cols[idx + 1:]:
                        if c2 not in cls:
                            continue
                        s2 = cls[c2]
                        # Check compatibility
                        ok = True
                        for e in s2:
                            if conf[e] & s1:
                                ok = False
                                break
                        if ok:
                            for e in s2:
                                col[e] = c1
                            s1 |= s2
                            del cls[c2]
                            merged = True
                            improved = True
                            break
                    if merged:
                        break

            # --- single-edge moves: try to move an edge to a lower colour ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    moved = False
                    for _ in range(1000):
                        e = rng.randrange(m)
                        c = col[e]
                        # try to move to any lower colour
                        for new_c in range(c):
                            if new_c in cls and not (conf[e] & cls[new_c]):
                                col[e] = new_c
                                moved = True
                                break
                        if moved:
                            break
                    if moved:
                        improved = True

            # --- 2-swap moves: exchange colours of two edges ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    swapped = False
                    for _ in range(500):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        if e1 == e2:
                            continue
                        c1, c2 = col[e1], col[e2]
                        if c1 == c2:
                            continue
                        # Check validity
                        valid = True
                        for j in conf[e1]:
                            if col[j] == c2 and j != e2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if col[j] == c1 and j != e1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2] = c2, c1
                            swapped = True
                            break
                    if swapped:
                        improved = True

        return reindex(col)

    def simulated_annealing(col, conf, m, deadline, rng, iterations=100000):
        col = list(col)
        best = list(col)
        best_k = k_of(col)
        current = list(col)
        current_k = k_of(current)
        no_improve = 0
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            if no_improve > 5000:
                break
            cls = build_classes(current)
            cols = sorted(cls.keys())
            if len(cols) < 2:
                break
            e = rng.randrange(m)
            c = current[e]
            others = [c2 for c2 in cols if c2 != c]
            if not others:
                break
            new_c = rng.choice(others)
            if new_c not in cls or not (conf[e] & cls[new_c]):
                current[e] = new_c
                new_k = k_of(current)
                if new_k < current_k:
                    current_k = new_k
                    no_improve = 0
                    if new_k < best_k:
                        best_k = new_k
                        best = list(current)
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return reindex(best)

    results = []
    for idx, (n, d, seed) in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue
        conf = build_conflict(edges)
        cdeg = [len(c) for c in conf]
        order_base = list(range(m))
        deg_order = sorted(order_base, key=lambda i: -cdeg[i])

        best = None
        best_k = m + 1

        def consider(col):
            nonlocal best, best_k
            k = k_of(col)
            if k < best_k:
                best_k = k
                best = list(col)

        # a few deterministic seeds
        consider(greedy([None] * m, order_base, conf))
        consider(greedy([None] * m, deg_order, conf))
        consider(greedy([None] * m, sorted(order_base, key=lambda i: cdeg[i]), conf))

        rng = Random(999 + seed)
        remaining = 3 - idx
        restart_deadline = DEADLINE - 25.0 * remaining

        # Randomized restarts with greedy
        while time.time() < restart_deadline:
            order = list(order_base)
            rng.shuffle(order)
            consider(greedy([None] * m, order, conf))

        # Local compression on the best
        ls_deadline = min(restart_deadline + 20.0, DEADLINE - 20.0 * (remaining - 1))
        if best is not None:
            best = local_improve(best, conf, m, ls_deadline, rng)
            consider(best)
            # Run local improve again for a second pass
            if time.time() < ls_deadline:
                best = local_improve(best, conf, m, ls_deadline, rng)
                consider(best)

        # Simulated annealing for further improvement
        sa_deadline = DEADLINE - 20.0 * (remaining - 1)
        if best is not None and time.time() < sa_deadline:
            sa_result = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=100000)
            consider(sa_result)
            if time.time() < sa_deadline:
                sa_result2 = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=100000)
                consider(sa_result2)
            # Final local polish
            if best is not None and time.time() < sa_deadline:
                final_deadline = DEADLINE - 10.0 * (remaining - 1)
                best = local_improve(best, conf, m, final_deadline, rng)
                consider(best)

        if best is None:
            best = greedy([None] * m, order_base, conf)
        results.append(best)
    return results
