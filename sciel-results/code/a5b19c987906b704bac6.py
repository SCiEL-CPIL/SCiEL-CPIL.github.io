import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 87.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)  # 91

    # ------------------------------------------------------------------
    # Fast incremental scorer
    # ------------------------------------------------------------------
    def make_scoring(values):
        sv = values[:]
        cnt_sum = {}
        cnt_prod = {}
        distinct_sum = 0
        distinct_prod = 0

        def add(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 0:
                if which == 's':
                    distinct_sum += 1
                else:
                    distinct_prod += 1
            cnt[x] = c + 1

        def rem(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 1:
                del cnt[x]
                if which == 's':
                    distinct_sum -= 1
                else:
                    distinct_prod -= 1
            else:
                cnt[x] = c - 1

        def swap(out, inn):
            u, v = out
            s0 = sv[u] + sv[v]
            p0 = sv[u] * sv[v]
            rem(s0, cnt_sum, 's')
            rem(p0, cnt_prod, 'p')
            u, v = inn
            s1 = sv[u] + sv[v]
            p1 = sv[u] * sv[v]
            add(s1, cnt_sum, 's')
            add(p1, cnt_prod, 'p')

        def add_edge(e):
            u, v = e
            add(sv[u] + sv[v], cnt_sum, 's')
            add(sv[u] * sv[v], cnt_prod, 'p')

        def raw():
            return max(distinct_sum, distinct_prod)

        def init_from(edges):
            for e in edges:
                add_edge(e)

        return init_from, swap, add_edge, raw

    def full_score(values, edge_list):
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(values[u] + values[v])
            p.add(values[u] * values[v])
        return max(len(s), len(p))

    # ------------------------------------------------------------------
    # Value generation: focus on structures that create collisions
    # ------------------------------------------------------------------
    def gen_arithmetic_progression():
        """AP values create many sum collisions."""
        start_v = random.randint(1, 8)
        step = random.choice([1, 2, 3])
        vals = [start_v + i * step for i in range(n)]
        return vals

    def gen_clustered():
        """Values clustered in small range."""
        center = random.randint(3, 15)
        spread = random.randint(2, 8)
        seen = set()
        vals = []
        attempts = 0
        while len(vals) < n and attempts < 200:
            v = center + random.randint(-spread, spread)
            if 1 <= v <= 10000 and v not in seen:
                seen.add(v)
                vals.append(v)
            attempts += 1
        while len(vals) < n:
            v = random.randint(1, 30)
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    def gen_multiplicative():
        """Values with multiplicative structure for product collisions."""
        # Use multiples of a small number
        base = random.choice([2, 3, 4, 5, 6])
        seen = set()
        vals = []
        for i in range(1, 100):
            v = base * i
            if 1 <= v <= 10000 and v not in seen:
                seen.add(v)
                vals.append(v)
            if len(vals) == n:
                break
        while len(vals) < n:
            v = random.randint(1, 50)
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    def gen_mixed_small():
        """Mix of very small values."""
        base_vals = list(range(1, 30))
        random.shuffle(base_vals)
        return base_vals[:n]

    def gen_powers_rich():
        """Values with power-of-2 factors for product structure."""
        candidates = []
        for i in range(1, 100):
            for j in range(1, 10):
                v = i * (2 ** j)
                if 1 <= v <= 10000:
                    candidates.append(v)
        random.shuffle(candidates)
        seen = set()
        vals = []
        for v in candidates:
            if v not in seen:
                seen.add(v)
                vals.append(v)
            if len(vals) == n:
                break
        while len(vals) < n:
            v = random.randint(1, 50)
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    def generate_values():
        strategies = [gen_arithmetic_progression, gen_clustered, gen_multiplicative,
                      gen_mixed_small, gen_powers_rich]
        return random.choice(strategies)()

    # ------------------------------------------------------------------
    # Greedy edge selection
    # ------------------------------------------------------------------
    def greedy_edges(values):
        """Select 35 edges to minimize distinct sums/products."""
        best_chosen = None
        best_score = float('inf')

        # Multiple ordering strategies
        strategies = []

        # Sort by sum
        order1 = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        strategies.append(order1)

        # Sort by product
        order2 = sorted(all_edges, key=lambda e: values[e[0]] * values[e[1]])
        strategies.append(order2)

        # Sort by (sum, product) lexicographically
        order3 = sorted(all_edges, key=lambda e: (values[e[0]] + values[e[1]],
                                                   values[e[0]] * values[e[1]]))
        strategies.append(order3)

        # Random order
        order4 = all_edges[:]
        random.shuffle(order4)
        strategies.append(order4)

        for order in strategies:
            chosen_temp = []
            cnt_sum = {}
            cnt_prod = {}
            distinct_sum = 0
            distinct_prod = 0

            for e in order:
                if len(chosen_temp) == m:
                    break
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]

                if s not in cnt_sum:
                    distinct_sum += 1
                cnt_sum[s] = cnt_sum.get(s, 0) + 1

                if p not in cnt_prod:
                    distinct_prod += 1
                cnt_prod[p] = cnt_prod.get(p, 0) + 1

                chosen_temp.append(e)

            score = max(distinct_sum, distinct_prod)
            if score < best_score:
                best_score = score
                best_chosen = chosen_temp

        return best_chosen, best_score

    # ------------------------------------------------------------------
    # Edge polish with hill-climbing (deterministic improvement)
    # ------------------------------------------------------------------
    def polish_edges_hillclimb(values, edges, deadline):
        """Hill-climb on edges: try all possible swaps, take best improving."""
        init_from, swap, add_edge, raw = make_scoring(values)
        init_from(edges)
        cur = raw()
        best = cur
        best_edges = list(edges)
        edges_set = set(edges)
        no_improve_rounds = 0

        while time.time() < deadline:
            improved = False
            # Try each edge against all possible replacements
            for i in range(m):
                e_out = edges[i]
                best_in = None
                best_r = cur  # need strict improvement

                for e_in in all_edges:
                    if e_in in edges_set:
                        continue
                    swap(e_out, e_in)
                    r = raw()
                    swap(e_in, e_out)  # revert

                    if r < best_r:
                        best_r = r
                        best_in = e_in

                if best_in is not None:
                    swap(e_out, best_in)
                    edges[i] = best_in
                    edges_set.discard(e_out)
                    edges_set.add(best_in)
                    cur = best_r
                    improved = True
                    if cur < best:
                        best = cur
                        best_edges = list(edges)

            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= 2:
                    break
            else:
                no_improve_rounds = 0

        return best, best_edges

    # ------------------------------------------------------------------
    # Edge polish with random + hill-climb
    # ------------------------------------------------------------------
    def polish_edges_random(values, edges, steps, deadline):
        """Random edge swaps with incremental scoring."""
        init_from, swap, add_edge, raw = make_scoring(values)
        init_from(edges)
        cur = raw()
        best = cur
        best_edges = list(edges)
        edges_set = set(edges)

        for _ in range(steps):
            if time.time() > deadline:
                break

            i = random.randrange(m)
            e_out = edges[i]

            # Try several random candidates
            best_e_in = None
            best_r = float('inf')

            for _ in range(8):
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue

                swap(e_out, e_in)
                r = raw()

                if r < best_r:
                    best_r = r
                    best_e_in = e_in

                swap(e_in, e_out)

            if best_e_in is None:
                continue

            e_out, e_in = best_e_in
            # Wait, best_e_in is just the edge, not a tuple
            # Let me fix this
            pass

        # Simpler version
        init_from, swap, add_edge, raw = make_scoring(values)
        init_from(edges)
        cur = raw()
        best = cur
        best_edges = list(edges)
        edges_set = set(edges)

        for _ in range(steps):
            if time.time() > deadline:
                break

            i = random.randrange(m)
            e_out = edges[i]

            best_e_in = None
            best_r = float('inf')

            for _ in range(8):
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue

                swap(e_out, e_in)
                r = raw()

                if r < best_r:
                    best_r = r
                    best_e_in = e_in

                swap(e_in, e_out)

            if best_e_in is None:
                continue

            swap(e_out, best_e_in)
            edges[i] = best_e_in
            edges_set.discard(e_out)
            edges_set.add(best_e_in)
            cur = best_r

            if cur < best:
                best = cur
                best_edges = list(edges)

        return best, best_edges

    # ------------------------------------------------------------------
    # Value polish: try replacing each value with candidates
    # ------------------------------------------------------------------
    def polish_values(values, edges, steps, deadline):
        """Improve values while keeping edges fixed."""
        vals = list(values)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)

        for _ in range(steps):
            if time.time() > deadline:
                break

            idx = random.randrange(n)
            old = vals[idx]

            # Generate candidates
            candidates = set()
            # Small random values
            for _ in range(5):
                candidates.add(random.randint(1, 25))
            # Medium values
            for _ in range(3):
                candidates.add(random.randint(10, 60))
            # Related to other values
            for _ in range(4):
                other = random.randrange(n)
                if other != idx:
                    candidates.add(vals[other] + random.randint(-3, 3))
                    candidates.add(vals[other] * random.choice([2, 3]))
                    candidates.add(vals[other] // 2 if vals[other] >= 2 else 1)

            best_nv = None
            best_r = float('inf')

            for nv in candidates:
                if nv < 1 or nv > 10000 or nv == old or nv in vals:
                    continue

                vals[idx] = nv
                r = full_score(vals, edges)

                if r < best_r:
                    best_r = r
                    best_nv = nv

                vals[idx] = old

            if best_nv is None:
                continue

            vals[idx] = best_nv
            cur = best_r

            if cur < best:
                best = cur
                best_vals = list(vals)

        return best, best_vals

    # ------------------------------------------------------------------
    # Simulated annealing for values
    # ------------------------------------------------------------------
    def anneal_values(values, edges, t0, t1, steps, deadline):
        """Simulated annealing on values."""
        vals = list(values)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)

        t = t0
        for _ in range(steps):
            if time.time() > deadline:
                break

            t = t0 * (t1 / t0) ** (1.0 / max(1, steps))
            idx = random.randrange(n)
            old = vals[idx]

            # Generate candidate
            if random.random() < 0.5:
                nv = random.randint(1, 30)
            else:
                nv = random.randint(1, 80)

            if nv < 1 or nv > 10000 or nv == old or nv in vals:
                continue

            vals[idx] = nv
            new_score = full_score(vals, edges)

            if new_score < cur or random.random() < math.exp(-(new_score - cur) / max(t, 0.01)):
                cur = new_score
                if cur < best:
                    best = cur
                    best_vals = list(vals)
            else:
                vals[idx] = old

        return best, best_vals

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------
    best_score = float('inf')
    best_vals = None
    best_edges = None

    def consider(sc, vals, edges):
        nonlocal best_score, best_vals, best_edges
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(edges)

    # Phase 1: Multiple value initializations with greedy edges + polish
    n_starts = 25
    for s in range(n_starts):
        if time.time() > deadline - 10.0:
            break

        vals = generate_values()
        edges, sc = greedy_edges(vals)
        consider(sc, vals, edges)

        # Hill-climb edges
        sc_hc, edges_hc = polish_edges_hillclimb(vals, edges, deadline - 8.0)
        consider(sc_hc, vals, edges_hc)

        # Random edge polish
        sc_rp, edges_rp = polish_edges_random(vals, edges_hc, 5000, deadline - 7.0)
        consider(sc_rp, vals, edges_rp)

    # Phase 2: From best so far, do joint value+edge optimization
    if best_vals is not None and time.time() < deadline - 5.0:
        vals = list(best_vals)
        edges = list(best_edges)

        # Value polish
        sc_v, vals_v = polish_values(vals, edges, 20000, deadline - 4.0)
        consider(sc_v, vals_v, edges)

        # Edge hill-climb on improved values
        sc_e, edges_e = polish_edges_hillclimb(vals_v, edges, deadline - 3.0)
        consider(sc_e, vals_v, edges_e)

        # Anneal values
        sc_a, vals_a = anneal_values(vals_v, edges_e, 5.0, 0.1, 15000, deadline - 2.0)
        consider(sc_a, vals_a, edges_e)

        # Final edge polish
        sc_f, edges_f = polish_edges_hillclimb(vals_a, edges_e, deadline - 1.0)
        consider(sc_f, vals_a, edges_f)

    # Phase 3: Final polish
    if best_vals is not None and time.time() < deadline - 1.0:
        sc_final, edges_final = polish_edges_hillclimb(best_vals, best_edges, deadline - 0.5)
        consider(sc_final, best_vals, edges_final)

    # Fallback
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)

    # ------------------------------------------------------------------
    # Strict type conversion and validation
    # ------------------------------------------------------------------
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]

    # Ensure exactly 14 distinct positive integers
    seen_vals = set()
    clean_vals = []
    for v in final_values:
        if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10000 and v not in seen_vals:
            seen_vals.add(v)
            clean_vals.append(v)
    while len(clean_vals) < n:
        v = random.randint(1, 10000)
        if v not in seen_vals:
            seen_vals.add(v)
            clean_vals.append(v)
    final_values = clean_vals[:n]

    # Ensure exactly 35 unique simple edges
    seen = set()
    clean_edges = []
    for e in final_edges:
        a, b = int(e[0]), int(e[1])
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        t = tuple(sorted((a, b)))
        if t in seen:
            continue
        seen.add(t)
        clean_edges.append([a, b])
    if len(clean_edges) < m:
        for e in all_edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                clean_edges.append([e[0], e[1]])
                if len(clean_edges) == m:
                    break
    final_edges = clean_edges[:m]

    return {'values': final_values, 'edges': final_edges}
