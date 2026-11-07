import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0   # safety margin vs 90s wall-clock kill
    rng = random.Random(12345)

    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def anneal_edges(values, budget):
        """Simulated annealing over the 35-edge set. Structural change vs greedy."""
        # Precompute per-edge sum/product
        es = [values[u] + values[v] for u, v in all_edges]
        ep = [values[u] * values[v] for u, v in all_edges]

        # Start from a good greedy seed (reuses existing machinery)
        selected = greedy_select_edges(values)
        sel_set = set(selected)

        # Build reverse maps: sum -> edges, prod -> edges
        sum_to_edges = {}
        prod_to_edges = {}
        for i in range(num_edges):
            sum_to_edges.setdefault(es[i], []).append(i)
            prod_to_edges.setdefault(ep[i], []).append(i)

        cur_sums = set(es[i] for i in sel_set)
        cur_prods = set(ep[i] for i in sel_set)
        cur_score = max(len(cur_sums), len(cur_prods))
        best_score = cur_score
        best_sel = set(sel_set)

        T = 30.0
        Tmin = 0.02
        step = 0.9992
        t = T

        while t > Tmin and time.time() < budget:
            # candidate: swap one in, one out
            i_out = rng.choice(list(sel_set))
            s_out = es[i_out]
            p_out = ep[i_out]
            # pick i_in that shares sum or product with something (high collision)
            # try a few random candidates, prefer those reusing existing sum/prod
            cand = None
            for _ in range(6):
                c = rng.randrange(num_edges)
                if c in sel_set:
                    continue
                cand = c
                break
            if cand is None:
                t *= step
                continue
            s_in = es[cand]
            p_in = ep[cand]

            new_sums = set(cur_sums)
            new_prods = set(cur_prods)
            # remove out
            # recompute membership counts cheaply
            # count occurrences of s_out among selected
            cnt_s_out = sum(1 for i in sel_set if es[i] == s_out)
            cnt_p_out = sum(1 for i in sel_set if ep[i] == p_out)
            if cnt_s_out == 1:
                new_sums.discard(s_out)
            if cnt_p_out == 1:
                new_prods.discard(p_out)
            new_sums.add(s_in)
            new_prods.add(p_in)
            new_score = max(len(new_sums), len(new_prods))

            delta = new_score - cur_score
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / t)):
                sel_set.remove(i_out)
                sel_set.add(cand)
                cur_sums = new_sums
                cur_prods = new_prods
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_sel = set(sel_set)
            t *= step

        return list(best_sel)

    def greedy_select_edges(values):
        """Greedy + local swap improvement (retained refinement machinery)."""
        es = [values[u] + values[v] for u, v in all_edges]
        ep = [values[u] * values[v] for u, v in all_edges]

        selected = []
        sel_set = set()
        cur_sums = set()
        cur_prods = set()
        remaining = list(range(num_edges))

        for _ in range(m):
            best_idx = None
            best_key = None
            for idx in remaining:
                add_s = 1 if es[idx] not in cur_sums else 0
                add_p = 1 if ep[idx] not in cur_prods else 0
                score = max(len(cur_sums) + add_s, len(cur_prods) + add_p)
                tie = add_s + add_p
                key = (score, tie, es[idx] + ep[idx])
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            selected.append(best_idx)
            sel_set.add(best_idx)
            cur_sums.add(es[best_idx])
            cur_prods.add(ep[best_idx])
            remaining.remove(best_idx)

        # local swap improvement
        improved = True
        while improved:
            improved = False
            cur_sums = set(es[i] for i in selected)
            cur_prods = set(ep[i] for i in selected)
            cur_score = max(len(cur_sums), len(cur_prods))
            for pos in range(m):
                old = selected[pos]
                # temp remove
                cnt_s = sum(1 for i in selected if es[i] == es[old])
                cnt_p = sum(1 for i in selected if ep[i] == ep[old])
                ts = set(cur_sums); tp = set(cur_prods)
                if cnt_s == 1: ts.discard(es[old])
                if cnt_p == 1: tp.discard(ep[old])
                for new in remaining:
                    ns = ts | {es[new]}
                    np_ = tp | {ep[new]}
                    if max(len(ns), len(np_)) < cur_score:
                        selected[pos] = new
                        remaining.remove(new)
                        remaining.append(old)
                        cur_score = max(len(ns), len(np_))
                        improved = True
                        break
                if improved:
                    break
        return selected

    # --- value generation: multiplicative structure to force product collisions ---
    def gen_candidates():
        cands = []
        # consecutive
        for start in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
            cands.append(list(range(start, start + n)))
        # arithmetic multiples
        for base in [2, 3, 4, 5, 6, 8]:
            vals = []
            v = base
            while len(vals) < n and v <= max_val:
                if v not in vals:
                    vals.append(v)
                v += base
            if len(vals) == n:
                cands.append(vals)
        # geometric-ish (powers) for product collisions
        for base in [2, 3]:
            vals = [base ** k for k in range(n)]
            if all(v <= max_val for v in vals):
                cands.append(vals)
        # clustered two blocks
        cands.append(list(range(1, 8)) + list(range(10, 17)))
        cands.append(list(range(1, 6)) + list(range(8, 15)))
        cands.append(list(range(2, 9)) + list(range(12, 19)))
        cands.append(list(range(1, 10)) + list(range(20, 28)))
        # shared-factor cluster: multiples of 6 and 7 to collide products
        cands.append([6 * i for i in range(1, n + 1)])
        cands.append([i * (i + 1) for i in range(1, n + 1)])
        # near-duplicate small set
        cands.append(list(range(1, 15)))
        return cands

    def dedup(cands):
        seen = set()
        out = []
        for vals in cands:
            if len(set(vals)) != n:
                continue
            if any(v < 1 or v > max_val for v in vals):
                continue
            key = tuple(sorted(vals))
            if key not in seen:
                seen.add(key)
                out.append(vals)
        return out

    cands = dedup(gen_candidates())

    best_score = float('inf')
    best_values = None
    best_edges = None

    for vals in cands:
        if time.time() > deadline - 8:
            break
        # try both greedy and anneal, keep the better
        e1 = greedy_select_edges(vals)
        s1 = evaluate(vals, e1)
        e2 = anneal_edges(vals, deadline - 6)
        s2 = evaluate(vals, e2)
        if s2 < s1:
            s1, e1 = s2, e2
        if s1 < best_score:
            best_score = s1
            best_values = vals
            best_edges = e1
            if best_score <= 12:
                break

    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = greedy_select_edges(best_values)
        best_score = evaluate(best_values, best_edges)

    # --- polish: value perturbation + re-optimize edges with anneal ---
    polish_deadline = deadline - 2
    while time.time() < polish_deadline:
        idx = rng.randint(0, n - 1)
        old_val = best_values[idx]
        tried = False
        for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -15, 15, -20, 20]:
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val or new_val in best_values:
                continue
            nv = best_values.copy()
            nv[idx] = new_val
            e1 = greedy_select_edges(nv)
            s1 = evaluate(nv, e1)
            e2 = anneal_edges(nv, polish_deadline - 1)
            s2 = evaluate(nv, e2)
            if s2 < s1:
                s1, e1 = s2, e2
            if s1 < best_score:
                best_score = s1
                best_values = nv
                best_edges = e1
                tried = True
                break
        if not tried:
            new_val = rng.randint(1, 60)
            if new_val == old_val or new_val in best_values:
                continue
            nv = best_values.copy()
            nv[idx] = new_val
            e1 = greedy_select_edges(nv)
            s1 = evaluate(nv, e1)
            e2 = anneal_edges(nv, polish_deadline - 1)
            s2 = evaluate(nv, e2)
            if s2 < s1:
                s1, e1 = s2, e2
            if s1 < best_score:
                best_score = s1
                best_values = nv
                best_edges = e1

    # ensure exactly 35 edges, valid
    edges = [list(e) for e in best_edges]
    if len(edges) != m:
        # fallback
        best_edges = greedy_select_edges(best_values)
        edges = [list(e) for e in best_edges]

    return {
        'values': [int(v) for v in best_values],
        'edges': edges
    }
