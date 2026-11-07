import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)  # 91
    all_edges_set = set(all_edges)

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

        def get_counts():
            return cnt_sum, cnt_prod, distinct_sum, distinct_prod

        return init_from, swap, add_edge, raw, get_counts

    def full_score(values, edge_list):
        s = set()
        p = set()
        vals = values
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))

    # ------------------------------------------------------------------
    # Value generation strategies
    # ------------------------------------------------------------------
    def value_pools():
        pools = []
        # Dense small ranges
        for hi in (18, 22, 28, 36, 48, 64, 90, 140, 220, 350, 500):
            pools.append(list(range(1, hi + 1)))

        # Arithmetic progressions
        for start_v in range(1, 30):
            for step in [2, 3, 4, 5, 6, 7, 8, 10]:
                pool = [start_v + i * step for i in range(50) if start_v + i * step <= 500]
                if len(pool) >= n:
                    pools.append(pool)

        # Geometric-like (powers and multiples)
        for base in [2, 3, 4, 5, 6]:
            pool = []
            v = base
            while v <= 2000 and len(pool) < 50:
                pool.append(v)
                v *= base
            if len(pool) >= n:
                pools.append(pool)

        # Composites rich
        base_sets = [
            [i * j for i in range(2, 20) for j in range(2, 20) if i * j <= 500]
        ]
        for bs in base_sets:
            unique = list(set(bs))
            unique.sort()
            if len(unique) >= n:
                pools.append(unique)

        # Multiples of small numbers
        for mult in [2, 3, 4, 5, 6, 8, 10, 12]:
            pool = [mult * i for i in range(1, 100) if mult * i <= 1000]
            if len(pool) >= n:
                pools.append(pool)

        return pools

    def random_values(pool):
        if len(pool) >= n:
            return random.sample(pool, n)
        vals = []
        seen = set()
        while len(vals) < n:
            v = random.randint(1, 10000)
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    def generate_structured_values():
        strategies = []

        # Strategy 1: Small consecutive values
        for start_v in range(1, 15):
            for span in range(14, 30):
                pool = list(range(start_v, start_v + span))
                if len(pool) >= n:
                    strategies.append(random.sample(pool, n))

        # Strategy 2: Values with additive structure (a, a+d, a+2d, ...)
        for a in range(1, 20):
            for d in range(1, 15):
                vals = [a + i * d for i in range(n)]
                if max(vals) <= 10000 and len(set(vals)) == n:
                    strategies.append(vals)

        # Strategy 3: Values with multiplicative structure
        for base in range(2, 10):
            for exp_range in [(0, 7), (1, 8), (2, 9)]:
                vals = [base ** e for e in range(exp_range[0], exp_range[1])]
                if len(vals) >= n and max(vals) <= 10000:
                    strategies.append(random.sample(vals, n))

        # Strategy 4: Mix of small and structured
        for _ in range(30):
            base = random.randint(1, 8)
            d = random.randint(1, 5)
            vals = []
            seen = set()
            for i in range(n):
                if random.random() < 0.6:
                    v = base + random.randint(0, 4) * d
                else:
                    v = random.randint(1, 30)
                while v in seen or v < 1 or v > 10000:
                    v = random.randint(1, 50)
                seen.add(v)
                vals.append(v)
            strategies.append(vals)

        # Strategy 5: Pairs that sum to same value
        for target_sum in range(20, 60):
            vals = []
            seen = set()
            for i in range(n // 2):
                a = random.randint(1, target_sum - 2)
                b = target_sum - a
                if a not in seen and b not in seen and a != b and a >= 1 and b >= 1:
                    vals.append(a)
                    vals.append(b)
                    seen.add(a)
                    seen.add(b)
            while len(vals) < n:
                v = random.randint(1, 40)
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
            if len(set(vals)) == n:
                strategies.append(vals)

        # Strategy 6: Pairs that multiply to same value
        for target_prod in range(30, 200):
            vals = []
            seen = set()
            for i in range(n // 2):
                a = random.randint(2, int(math.sqrt(target_prod)) + 1)
                if target_prod % a == 0:
                    b = target_prod // a
                    if a not in seen and b not in seen and a != b:
                        vals.append(a)
                        vals.append(b)
                        seen.add(a)
                        seen.add(b)
            while len(vals) < n:
                v = random.randint(1, 40)
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
            if len(set(vals)) == n:
                strategies.append(vals)

        return strategies

    # ------------------------------------------------------------------
    # Greedy edge selection
    # ------------------------------------------------------------------
    def greedy_edges(values):
        chosen = []
        chosen_set = set()

        # Try multiple ordering strategies
        max_val = max(values)
        strategies = [
            sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]]),
            sorted(all_edges, key=lambda e: values[e[0]] * values[e[1]]),
            sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]] +
                   (values[e[0]] * values[e[1]]) / (max_val * max_val + 1)),
            sorted(all_edges, key=lambda e: abs(values[e[0]] - values[e[1]])),
            sorted(all_edges, key=lambda e: values[e[0]] * values[e[1]] /
                   (values[e[0]] + values[e[1]] + 1)),
        ]

        best_chosen = None
        best_score = float('inf')

        for order in strategies:
            chosen_temp = []
            chosen_set_temp = set()
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
                chosen_set_temp.add(e)

            score = max(distinct_sum, distinct_prod)
            if score < best_score:
                best_score = score
                best_chosen = chosen_temp
                best_chosen_set = chosen_set_temp

        return best_chosen, best_chosen_set

    # ------------------------------------------------------------------
    # Edge-only local search
    # ------------------------------------------------------------------
    def polish_edges(values, chosen, chosen_set, steps, deadline):
        init_from, swap, add_edge, raw, get_counts = make_scoring(values)
        init_from(chosen)
        cur = raw()
        best = cur
        no_improve = 0

        for _ in range(steps):
            if time.time() > deadline:
                break

            i = random.randrange(m)
            e_out = chosen[i]

            best_e_in = None
            best_r = float('inf')
            best_swap_data = None

            for _ in range(8):
                e_in = random.choice(all_edges)
                if e_in in chosen_set:
                    continue

                swap(e_out, e_in)
                r = raw()

                if r < best_r:
                    best_r = r
                    best_e_in = e_in
                    best_swap_data = (e_out, e_in)

                swap(e_in, e_out)

            if best_e_in is None:
                no_improve += 1
                if no_improve > 10000:
                    break
                continue

            e_out, e_in = best_swap_data
            swap(e_out, e_in)
            r = raw()

            if r <= cur:
                cur = r
                chosen[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                if r < best:
                    best = r
                no_improve = 0
            else:
                swap(e_in, e_out)
                no_improve += 1

        return chosen, chosen_set, best

    # ------------------------------------------------------------------
    # Joint value+edge local search
    # ------------------------------------------------------------------
    def joint_search(vals0, edges0, steps, deadline):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, get_counts = make_scoring(vals)
        init_from(edges)
        no_improve = 0

        for _ in range(steps):
            if time.time() > deadline:
                break

            if random.random() < 0.5:
                # edge move (incremental)
                i = random.randrange(m)
                e_out = edges[i]

                best_e_in = None
                best_r = float('inf')
                best_swap_data = None

                for _ in range(5):
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue

                    swap(e_out, e_in)
                    r = raw()

                    if r < best_r:
                        best_r = r
                        best_e_in = e_in
                        best_swap_data = (e_out, e_in)

                    swap(e_in, e_out)

                if best_e_in is None:
                    no_improve += 1
                    continue

                e_out, e_in = best_swap_data
                swap(e_out, e_in)
                r = raw()

                if r <= cur:
                    cur = r
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    swap(e_in, e_out)
                    no_improve += 1
            else:
                # value move (recompute)
                idx = random.randrange(n)
                old = vals[idx]

                candidates = []
                # Small values
                for _ in range(4):
                    candidates.append(random.randint(1, 40))
                # Medium values
                for _ in range(2):
                    candidates.append(random.randint(10, 80))
                # Related to existing values
                for _ in range(3):
                    other_idx = random.randrange(n)
                    if other_idx != idx:
                        candidates.append(vals[other_idx] + random.randint(-5, 5))
                        candidates.append(vals[other_idx] * random.choice([2, 3]))
                        candidates.append(vals[other_idx] // random.choice([2, 3]) if vals[other_idx] >= 2 else 1)

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
                    no_improve += 1
                    continue

                vals[idx] = best_nv
                r = best_r

                if r <= cur:
                    cur = r
                    init_from, swap, add_edge, raw, get_counts = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    vals[idx] = old
                    no_improve += 1

            if no_improve > 20000:
                break

        return best, best_vals, best_edges

    # ------------------------------------------------------------------
    # Driver
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

    # Phase 1: structured value pools -> greedy edges -> polish edges.
    pools = value_pools()
    random.shuffle(pools)
    structured_starts = generate_structured_values()

    # Process pools
    for pool in pools:
        if time.time() > deadline - 10.0:
            break
        for _ in range(4):
            if time.time() > deadline - 9.0:
                break
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                               10000, deadline - 6.0)
            consider(psc, vals, polished)

    # Process structured starts
    for vals in structured_starts:
        if time.time() > deadline - 6.0:
            break
        chosen, chosen_set = greedy_edges(vals)
        sc = full_score(vals, chosen)
        consider(sc, vals, chosen)
        polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                           8000, deadline - 4.0)
        consider(psc, vals, polished)

    # Phase 2: joint search from several starts.
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))

    for _ in range(10):
        if random.random() < 0.5:
            vals = random_values(pools[random.randrange(len(pools))])
        else:
            vals = random.choice(structured_starts)
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))

    for vals in structured_starts[:8]:
        chosen, chosen_set = greedy_edges(vals)
        starts.append((list(vals), list(chosen)))

    remaining = deadline - 8.0 - time.time()
    if remaining < 2.0:
        remaining = 2.0
    per = max(8000, int(remaining / max(1, len(starts)) / 0.0002))

    for vals0, edges0 in starts:
        if time.time() > deadline - 8.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 5.0)
        consider(sc, vals, edges)

    # Phase 3: final edge polish for best values.
    if time.time() < deadline - 5.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           80000, deadline - 3.0)
        consider(psc, best_vals, polished)

    # Phase 4: final value polish for best edges.
    if time.time() < deadline - 3.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        no_improve = 0

        while it < 50000 and time.time() < deadline - 2.0:
            it += 1
            idx = random.randrange(n)
            old = vals[idx]

            candidates = []
            for _ in range(5):
                candidates.append(random.randint(1, 50))
            for _ in range(3):
                candidates.append(random.randint(10, 120))
            for _ in range(3):
                other_idx = random.randrange(n)
                if other_idx != idx:
                    candidates.append(vals[other_idx] + random.randint(-5, 5))
                    candidates.append(vals[other_idx] * random.choice([2, 3]))
                    candidates.append(vals[other_idx] // random.choice([2, 3]) if vals[other_idx] >= 2 else 1)

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
                no_improve += 1
                if no_improve > 15000:
                    break
                continue

            vals[idx] = best_nv
            r = best_r

            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
                no_improve = 0
            else:
                vals[idx] = old
                no_improve += 1

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)

    # Final verification and cleanup
    seen_vals = set()
    clean_vals = []
    for v in best_vals:
        if v not in seen_vals and 1 <= v <= 10000:
            seen_vals.add(v)
            clean_vals.append(v)

    while len(clean_vals) < n:
        v = random.randint(1, 10000)
        if v not in seen_vals:
            seen_vals.add(v)
            clean_vals.append(v)

    best_vals = clean_vals[:n]

    # Strict type conversion
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    seen = set()
    clean = []
    for e in final_edges:
        a, b = e
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        t = tuple(sorted((a, b)))
        if t in seen:
            continue
        seen.add(t)
        clean.append([a, b])
    if len(clean) < m:
        for e in all_edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                clean.append([e[0], e[1]])
                if len(clean) == m:
                    break
    final_edges = clean[:m]

    return {'values': final_values, 'edges': final_edges}
