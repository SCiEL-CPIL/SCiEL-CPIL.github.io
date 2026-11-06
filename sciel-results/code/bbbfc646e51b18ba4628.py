"""Strong edge colouring: build the conflict graph, then use a fast simulated-annealing
style local search that (a) tries to recolour individual edges to free a colour class,
and (b) merges colour classes when compatible.  We run several restarts with different
random orderings and keep the best solution.  A hard deadline guard ensures we return
in time.
"""
import time
from random import Random
import sys
import math
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0  # safety margin under the 150 s kill

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
        # Precompute conflict lists as sorted lists for faster iteration
        conf_list = [sorted(c) for c in conf]
        return conf_list

    def greedy(col, order, conf):
        m = len(order)
        col = [-1] * m
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
        cls = defaultdict(set)
        for i, c in enumerate(col):
            cls[c].add(i)
        return cls

    def reindex(col):
        present = sorted(set(col))
        if not present:
            return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def is_valid_move(col, e, new_c, conf):
        """Check if edge e can be assigned colour new_c."""
        for j in conf[e]:
            if col[j] == new_c:
                return False
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
            merged_any = False
            while time.time() < deadline:
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
                        # Check compatibility: no edge in s2 conflicts with any edge in s1
                        ok = True
                        for e in s2:
                            for j in conf[e]:
                                if j in s1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            for e in s2:
                                col[e] = c1
                            s1.update(s2)
                            del cls[c2]
                            merged = True
                            improved = True
                            merged_any = True
                            break
                    if merged:
                        break
                if not merged:
                    break

            # --- single-edge moves: try to move an edge to a lower colour ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    moved = False
                    for _ in range(3000):
                        e = rng.randrange(m)
                        c = col[e]
                        # try to move to any lower colour
                        for new_c in range(c):
                            if is_valid_move(col, e, new_c, conf):
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
                    for _ in range(2000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        if e1 == e2:
                            continue
                        c1, c2 = col[e1], col[e2]
                        if c1 == c2:
                            continue
                        # Check validity of swap
                        valid = True
                        for j in conf[e1]:
                            if j != e2 and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j != e1 and col[j] == c1:
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
        cls = build_classes(current)
        t = 1.0
        t_min = 0.01
        cooling = 0.99995
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            if no_improve > 12000:
                break
            
            cols = sorted(cls.keys())
            if len(cols) < 2:
                break
            
            # Weighted random selection: prefer edges in higher-numbered colour classes
            # to encourage moving them to lower colours
            weights = []
            for c in cols:
                for e in cls[c]:
                    weights.append((e, c))
            if not weights:
                break
            
            # Use weighted random selection favoring higher colour classes
            total_weight = sum(1 + c for _, c in weights)
            r = rng.random() * total_weight
            cum = 0
            selected_e, selected_c = weights[0]
            for e, c in weights:
                cum += 1 + c
                if r <= cum:
                    selected_e, selected_c = e, c
                    break
            e = selected_e
            c = selected_c
            
            # Try to move to a different colour, preferring lower colours
            others = [c2 for c2 in cols if c2 != c]
            if not others:
                break
            
            # Sort others by colour number (lower first) and pick with bias
            others_sorted = sorted(others)
            # Higher probability for lower colour numbers
            probs = [1.0 / (1.0 + c2) for c2 in others_sorted]
            total_p = sum(probs)
            r = rng.random() * total_p
            cum = 0
            new_c = others_sorted[0]
            for c2, p in zip(others_sorted, probs):
                cum += p
                if r <= cum:
                    new_c = c2
                    break
            
            if is_valid_move(current, e, new_c, conf):
                # Remove e from old class
                cls[c].discard(e)
                if not cls[c]:
                    del cls[c]
                # Add e to new class
                cls[new_c].add(e)
                current[e] = new_c
                
                new_k = k_of(current)
                delta_k = new_k - current_k
                if delta_k <= 0:
                    current_k = new_k
                    no_improve = 0
                    if new_k < best_k:
                        best_k = new_k
                        best = list(current)
                else:
                    # Accept with probability exp(-delta/t)
                    if rng.random() < math.exp(-delta_k / max(t, 1e-6)):
                        current_k = new_k
                    else:
                        # Revert
                        cls[new_c].discard(e)
                        cls[c].add(e)
                        current[e] = c
                    no_improve += 1
            else:
                no_improve += 1
            
            t = max(t * cooling, t_min)
        
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
        restart_deadline = DEADLINE - 15.0 * remaining

        # Randomized restarts with greedy
        while time.time() < restart_deadline:
            order = list(order_base)
            rng.shuffle(order)
            consider(greedy([None] * m, order, conf))

        # Local compression on the best
        ls_deadline = min(restart_deadline + 35.0, DEADLINE - 15.0 * (remaining - 1))
        if best is not None:
            best = local_improve(best, conf, m, ls_deadline, rng)
            consider(best)
            # Run local improve again for a second pass
            if time.time() < ls_deadline:
                best = local_improve(best, conf, m, ls_deadline, rng)
                consider(best)
            # Third pass if time permits
            if time.time() < ls_deadline:
                best = local_improve(best, conf, m, ls_deadline, rng)
                consider(best)
            # Fourth pass if time permits
            if time.time() < ls_deadline:
                best = local_improve(best, conf, m, ls_deadline, rng)
                consider(best)

        # Simulated annealing for further improvement
        sa_deadline = DEADLINE - 10.0 * (remaining - 1)
        if best is not None and time.time() < sa_deadline:
            sa_result = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=300000)
            consider(sa_result)
            if time.time() < sa_deadline:
                sa_result2 = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=300000)
                consider(sa_result2)
            if time.time() < sa_deadline:
                sa_result3 = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=300000)
                consider(sa_result3)
            # Final local polish with extended deadline
            if best is not None and time.time() < sa_deadline:
                final_deadline = DEADLINE - 5.0 * (remaining - 1)
                best = local_improve(best, conf, m, final_deadline, rng)
                consider(best)
                # One more polish pass
                if time.time() < final_deadline:
                    best = local_improve(best, conf, m, final_deadline, rng)
                    consider(best)

        if best is None:
            best = greedy([None] * m, order_base, conf)
        results.append(best)
    return results
