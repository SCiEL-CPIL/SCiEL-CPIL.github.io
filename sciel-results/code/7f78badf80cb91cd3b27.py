import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score(values, edge_list):
        """Fast score computation using local variables."""
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))

    def best_edges_for_values(values, budget):
        """Greedy + local search: pick 35 edges to minimize score, for fixed values."""
        # Greedy init: pick edges that reuse existing sum/product
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        # Sort all edges by a heuristic: prefer pairs with small values
        edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        for e in edge_order:
            if len(chosen) == m:
                break
            chosen.append(e)
            chosen_set.add(e)
            sums.add(values[e[0]] + values[e[1]])
            prods.add(values[e[0]] * values[e[1]])
        # Local search to improve with fast scoring
        cur = list(chosen)
        cur_score = fast_score(values, cur)
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            new_cur = list(cur)
            new_cur[i] = e_in
            new_score = fast_score(values, new_cur)
            if new_score <= cur_score:
                cur = new_cur
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000 and random.random() < 0.01:
                    no_improve = 0
        return cur, cur_score

    def best_values_for_edges(edges, budget):
        """Local search on values for fixed edges."""
        # Start with small distinct values
        vals = random.sample(range(1, 41), n)
        elist = list(edges)
        cur_score = fast_score(vals, elist)
        best = list(vals)
        best_score = cur_score
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            if random.random() < 0.5:
                # replace
                new_val = random.randint(1, 100)
            else:
                new_val = vals[idx] + random.randint(-15, 15)
            if not (1 <= new_val <= 10000) or new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = fast_score(new_vals, elist)
            if s < cur_score:
                vals = new_vals
                cur_score = s
                no_improve = 0
                if s < best_score:
                    best = list(vals)
                    best_score = s
            elif s == cur_score and random.random() < 0.1:
                vals = new_vals
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 3000:
                    no_improve = 0
        return best, best_score

    def joint_search(vals0, edges0, budget):
        """Alternate value and edge moves with targeted collision-seeking."""
        vals = list(vals0)
        edges = set(edges0)
        elist = list(edges)
        cur_score = fast_score(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(edges)
        it = 0
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            r = random.random()
            if r < 0.40:
                # edge move
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                new_elist = list(elist)
                new_elist[i] = e_in
                s = fast_score(vals, new_elist)
                if s <= cur_score:
                    elist = new_elist
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score = s
            elif r < 0.70:
                # value move: random
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 80)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = fast_score(new_vals, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            else:
                # targeted collision-seeking: pick a vertex and try to make
                # its edge sums/products collide with existing ones
                idx = random.randint(0, n - 1)
                # Find neighbors of idx in current edge set
                neighbors = [v for u, v in elist if u == idx or v == idx]
                if not neighbors:
                    continue
                # Get current sums and products involving idx
                existing_sums = set()
                existing_prods = set()
                for u, v in elist:
                    if u == idx or v == idx:
                        other = v if u == idx else u
                        existing_sums.add(vals[idx] + vals[other])
                        existing_prods.add(vals[idx] * vals[other])
                # Try to set vals[idx] so that for some neighbor, the sum
                # matches an existing sum or product matches an existing product
                candidates = set()
                for s_val in existing_sums:
                    for nb in neighbors:
                        cand = s_val - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                for p_val in existing_prods:
                    for nb in neighbors:
                        if vals[nb] != 0 and p_val % vals[nb] == 0:
                            cand = p_val // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                # Also try small random values
                candidates.add(random.randint(1, 50))
                if not candidates:
                    continue
                new_val = random.choice(list(candidates))
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = fast_score(new_vals, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    def try_direct_construction():
        """Direct construction: find values + edges with high collision."""
        best = None
        # Strategy: pick 14 values from a small pool, then find 35 edges.
        # Small pool -> small product range -> collisions.
        pools = [
            list(range(1, 25)),
            list(range(2, 30)),
            list(range(1, 20)),
            [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
            [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
            [2,3,4,5,6,7,8,9,10,11,12,13,14,16],
            [1,2,3,4,5,6,7,8,9,11,12,14,16,18],
        ]
        for pool in pools:
            for _ in range(3):
                vals = random.sample(pool, n)
                edges, sc = best_edges_for_values(vals, budget=3000)
                if best is None or sc < best[0]:
                    best = (sc, list(vals), edges)
                if time.time() > deadline - 30:
                    return best
        return best

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: direct construction
    d = try_direct_construction()
    if d is not None:
        best_score, best_vals, best_edges = d

    # Phase 2: joint search from multiple starts
    random.seed(1234)
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), set(best_edges)))
    # more random starts with small values
    for _ in range(8):
        vals = random.sample(range(1, 60), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    remaining = deadline - time.time() - 8.0
    per = max(2000, int(remaining / len(starts) / 0.00005))
    for vals0, edges0 in starts:
        if time.time() > deadline - 8.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges

    # Phase 3: polish edges for best values
    if time.time() < deadline - 4.0:
        elist = list(best_edges)
        it = 0
        while it < 20000 and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            new_elist = list(elist)
            new_elist[i] = e_in
            s = fast_score(best_vals, new_elist)
            if s < best_score:
                elist = new_elist
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s

    # Phase 4: polish values for best edges with targeted moves
    if time.time() < deadline - 1.0:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 15000 and time.time() < deadline - 0.5:
            it += 1
            r = random.random()
            if r < 0.6:
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 100)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = fast_score(new_vals, elist)
                if s < best_score:
                    vals = new_vals
                    best_score = s
                    best_vals = new_vals
            else:
                # targeted: try to create a collision
                idx = random.randint(0, n - 1)
                neighbors = [v for u, v in elist if u == idx or v == idx]
                if not neighbors:
                    continue
                existing_sums = set()
                existing_prods = set()
                for u, v in elist:
                    if u == idx or v == idx:
                        other = v if u == idx else u
                        existing_sums.add(vals[idx] + vals[other])
                        existing_prods.add(vals[idx] * vals[other])
                candidates = set()
                for s_val in existing_sums:
                    for nb in neighbors:
                        cand = s_val - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                for p_val in existing_prods:
                    for nb in neighbors:
                        if vals[nb] != 0 and p_val % vals[nb] == 0:
                            cand = p_val // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                candidates.add(random.randint(1, 50))
                if not candidates:
                    continue
                new_val = random.choice(list(candidates))
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = fast_score(new_vals, elist)
                if s < best_score:
                    vals = new_vals
                    best_score = s
                    best_vals = new_vals

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
