"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def _delta(values, edges):
    """Return (max(#distinct sums,#distinct products), sums_set, prods_set)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods)), sums, prods

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    n = P['n']
    m = P['m']
    maximum = P['maximum']

    random.seed(1234)
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def greedy_edges(values):
        """Pick m edges minimizing max(#distinct sums,#distinct products)."""
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                s = values[i] + values[j]
                p = values[i] * values[j]
                all_pairs.append((i, j, s, p))

        sum_counts = {}
        prod_counts = {}
        for _, _, s, p in all_pairs:
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1

        remaining = list(all_pairs)
        remaining.sort(key=lambda x: min(sum_counts[x[2]], prod_counts[x[3]]), reverse=True)

        used_sums = set()
        used_prods = set()
        selected = []
        for _ in range(m):
            best_k = -1
            best_score = float('inf')
            for k in range(len(remaining)):
                i, j, s, p = remaining[k]
                ns = len(used_sums) + (0 if s in used_sums else 1)
                np_ = len(used_prods) + (0 if p in used_prods else 1)
                score = max(ns, np_)
                if score < best_score:
                    best_score = score
                    best_k = k
            if best_k < 0:
                break
            i, j, s, p = remaining[best_k]
            selected.append([i, j])
            used_sums.add(s)
            used_prods.add(p)
            remaining.pop(best_k)
        return selected, used_sums, used_prods

    def local_search(values, edges):
        """Mutate values and swap edges, accepting improvements (fast incremental)."""
        best_score, sums, prods = _delta(values, edges)
        edges = [list(e) for e in edges]
        values = list(values)
        edge_set = set(tuple(e) for e in edges)
        while time.time() < deadline - 2.0:
            improved = False
            # --- value mutations ---
            for _ in range(30):
                if time.time() > deadline - 2.0:
                    break
                idx = random.randrange(n)
                old = values[idx]
                cand = random.randint(1, maximum)
                if cand == old or cand in values:
                    continue
                values[idx] = cand
                ns = len(sums)
                np_ = len(prods)
                for k, (a, b) in enumerate(edges):
                    if a == idx or b == idx:
                        o = edges[k]
                        os_ = values[a] + values[b]
                        op = values[a] * values[b]
                        ns -= 1 if (sums[os_] == 1) else 0
                        np_ -= 1 if (prods[op] == 1) else 0
                        ns += 1 if os_ not in sums else 0
                        np_ += 1 if op not in prods else 0
                        ns_ = values[a] + values[b]
                        npd = values[a] * values[b]
                        sums[ns_] = sums.get(ns_, 0) + 1
                        prods[npd] = prods.get(npd, 0) + 1
                        ns += 1 if sums[ns_] == 1 else 0
                        np_ += 1 if prods[npd] == 1 else 0
                        sums[os_] -= 1
                        prods[op] -= 1
                        if sums[os_] == 0:
                            del sums[os_]
                        if prods[op] == 0:
                            del prods[op]
                if max(ns, np_) < best_score:
                    best_score = max(ns, np_)
                    improved = True
                else:
                    values[idx] = old
                    best_score, sums, prods = _delta(values, edges)
            # --- edge swaps ---
            for _ in range(30):
                if time.time() > deadline - 2.0:
                    break
                pos = random.randrange(m)
                old_edge = tuple(edges[pos])
                new_edge = random.choice(all_edges)
                if new_edge in edge_set:
                    continue
                a, b = new_edge
                os_ = values[old_edge[0]] + values[old_edge[1]]
                op = values[old_edge[0]] * values[old_edge[1]]
                ns_ = values[a] + values[b]
                npd = values[a] * values[b]
                ns = len(sums)
                np_ = len(prods)
                ns -= 1 if sums[os_] == 1 else 0
                np_ -= 1 if prods[op] == 1 else 0
                ns += 1 if ns_ not in sums else 0
                np_ += 1 if npd not in prods else 0
                sums[os_] -= 1
                prods[op] -= 1
                if sums[os_] == 0:
                    del sums[os_]
                if prods[op] == 0:
                    del prods[op]
                sums[ns_] = sums.get(ns_, 0) + 1
                prods[npd] = prods.get(npd, 0) + 1
                ns += 1 if sums[ns_] == 1 else 0
                np_ += 1 if prods[npd] == 1 else 0
                if max(ns, np_) < best_score:
                    best_score = max(ns, np_)
                    edges[pos] = list(new_edge)
                    edge_set.remove(old_edge)
                    edge_set.add(new_edge)
                    improved = True
                else:
                    sums[ns_] -= 1
                    prods[npd] -= 1
                    if sums[ns_] == 0:
                        del sums[ns_]
                    if prods[npd] == 0:
                        del prods[npd]
                    sums[os_] = sums.get(os_, 0) + 1
                    prods[op] = prods.get(op, 0) + 1
            if not improved:
                # perturb: try a few random value changes to escape local minima
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    idx = random.randrange(n)
                    old = values[idx]
                    cand = random.randint(1, maximum)
                    if cand in values:
                        continue
                    values[idx] = cand
                    best_score, sums, prods = _delta(values, edges)
                    if best_score >= 40:
                        values[idx] = old
                        best_score, sums, prods = _delta(values, edges)
                        break
        return values, edges, best_score

    def gen_value_set(attempt):
        if attempt % 5 == 0:
            a = random.randint(1, 60)
            d = random.randint(1, 40)
            vals = [a + i * d for i in range(n)]
            if max(vals) > maximum:
                vals = list(range(1, n + 1))
        elif attempt % 5 == 1:
            base = random.randint(1, 15)
            vals = [base + random.randint(0, 30) for _ in range(n)]
        elif attempt % 5 == 2:
            vals = random.sample(range(1, maximum + 1), n)
        elif attempt % 5 == 3:
            small = random.sample(range(1, 80), n // 2)
            large = random.sample(range(500, maximum + 1), n - n // 2)
            vals = small + large
        else:
            # near-arithmetic with jitter
            a = random.randint(1, 30)
            d = random.randint(1, 25)
            vals = [a + i * d + random.randint(-2, 2) for i in range(n)]
        # enforce distinct positive ints <= maximum
        seen = set()
        out = []
        for v in vals:
            if 1 <= v <= maximum and v not in seen:
                seen.add(v)
                out.append(v)
        while len(out) < n:
            v = random.randint(1, maximum)
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out[:n]

    best_raw = float('inf')
    best_values = None
    best_edges = None

    n_restarts = 25
    for attempt in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        vals = gen_value_set(attempt)
        edges, _, _ = greedy_edges(vals)
        vals, edges, score = local_search(vals, edges)
        if score < best_raw:
            best_raw = score
            best_values = vals
            best_edges = edges

    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_edges[:m]]

    return {'values': [int(v) for v in best_values],
            'edges': [[int(i), int(j)] for i, j in best_edges]}
# EVOLVE-BLOCK-END
