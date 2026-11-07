import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 86.0

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

    def score_delta(values, edge_list, e_out, e_in):
        """Compute new score after removing e_out and adding e_in."""
        u_out, v_out = e_out
        u_in, v_in = e_in
        s_out = values[u_out] + values[v_out]
        p_out = values[u_out] * values[v_out]
        s_in = values[u_in] + values[u_in] if False else values[u_in] + values[v_in]
        p_in = values[u_in] * values[v_in]
        
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        sum_counts[s_out] -= 1
        if sum_counts[s_out] <= 0:
            del sum_counts[s_out]
        prod_counts[p_out] -= 1
        if prod_counts[p_out] <= 0:
            del prod_counts[p_out]
        
        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
        
        return max(len(sum_counts), len(prod_counts))

    def collision_density(values, edge_list):
        """Compute collision density: for each edge, how many other edges share its sum or product."""
        sums = {}
        prods = {}
        for e in edge_list:
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]
            sums[s] = sums.get(s, 0) + 1
            prods[p] = prods.get(p, 0) + 1
        density = {}
        for e in edge_list:
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]
            density[e] = (sums[s] - 1) + (prods[p] - 1)
        return density

    def best_edges_for_values(values, budget):
        """Greedy + local search: pick 35 edges to minimize score, for fixed values."""
        # Greedy init: pick edges that maximize collision with already-chosen edges
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        
        # First pass: compute all edge sums/prods
        edge_props = {}
        for e in all_edges:
            u, v = e
            edge_props[e] = (values[u] + values[v], values[u] * values[v])
        
        # Greedy: at each step, pick the edge that creates the most collisions
        for _ in range(m):
            best_e = None
            best_collisions = -1
            for e in all_edges:
                if e in chosen_set:
                    continue
                s, p = edge_props[e]
                coll = (1 if s in sums else 0) + (1 if p in prods else 0)
                # Also consider potential for future collisions
                if coll > best_collisions:
                    best_collisions = coll
                    best_e = e
            if best_e is None:
                # Fallback: pick random unchosen edge
                remaining = [e for e in all_edges if e not in chosen_set]
                best_e = random.choice(remaining)
            chosen.append(best_e)
            chosen_set.add(best_e)
            s, p = edge_props[best_e]
            sums.add(s)
            prods.add(p)
        
        cur = list(chosen)
        cur_score = score_of(values, cur)
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            new_score = score_delta(values, cur, e_out, e_in)
            if new_score <= cur_score:
                cur[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000 and random.random() < 0.01:
                    no_improve = 0
        return cur, cur_score

    def multiplicative_values():
        """Generate values with multiplicative structure to induce product collisions."""
        # Use products of small primes/factors
        base = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32], 14)
        # Ensure distinctness
        while len(set(base)) < 14:
            base.append(random.randint(1, 50))
        base = list(set(base))[:14]
        random.shuffle(base)
        return base

    def joint_search(vals0, edges0, budget):
        """Alternate value and edge moves with fast scoring."""
        vals = list(vals0)
        edges = set(edges0)
        elist = list(edges)
        cur_score = score_of(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(edges)
        it = 0
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            if random.random() < 0.5:
                # edge move - use fast delta scoring
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                s = score_delta(vals, elist, e_out, e_in)
                if s <= cur_score:
                    elist[i] = e_in
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score = s
            else:
                # value move - try multiplicative structures
                idx = random.randint(0, n - 1)
                r = random.random()
                if r < 0.3:
                    # Small random value
                    new_val = random.randint(1, 40)
                elif r < 0.6:
                    # Multiplicative: product of two small numbers
                    a = random.randint(1, 12)
                    b = random.randint(1, 12)
                    new_val = a * b
                elif r < 0.8:
                    # Power of small base
                    base = random.choice([2, 3, 4, 5])
                    exp = random.randint(1, 4)
                    new_val = base ** exp
                else:
                    # Perturbation of current value
                    new_val = vals[idx] + random.randint(-10, 10)
                
                if not (1 <= new_val <= 10000) or new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, elist)
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
        # Strategy 1: Small pool values
        pools = [
            list(range(1, 25)),
            list(range(2, 30)),
            list(range(1, 20)),
            [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
            [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
            [1,2,3,4,5,6,8,9,10,12,15,16,18,20],
        ]
        for pool in pools:
            for _ in range(5):
                vals = random.sample(pool, n)
                edges, sc = best_edges_for_values(vals, budget=3000)
                if best is None or sc < best[0]:
                    best = (sc, list(vals), edges)
                if time.time() > deadline - 35:
                    return best
        
        # Strategy 2: Multiplicative structure values
        for _ in range(8):
            vals = multiplicative_values()
            edges, sc = best_edges_for_values(vals, budget=3000)
            if best is None or sc < best[0]:
                best = (sc, list(vals), edges)
            if time.time() > deadline - 30:
                return best
        
        return best

    def targeted_collision_search(vals, edges, budget):
        """Targeted search: find edge swaps that maximize collision density."""
        elist = list(edges)
        edge_set = set(edges)
        cur_score = score_of(vals, elist)
        best_score = cur_score
        best_elist = list(elist)
        best_edge_set = set(edge_set)
        
        # Precompute all edge properties
        edge_props = {}
        for e in all_edges:
            u, v = e
            edge_props[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        it = 0
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            # Try to find an edge that, when swapped in, reduces distinct sums or products
            i = random.randint(0, m - 1)
            e_out = elist[i]
            
            # Find candidate edges that share sum or product with existing edges
            s_out, p_out = edge_props[e_out]
            candidates = []
            for e_in in all_edges:
                if e_in in edge_set:
                    continue
                s_in, p_in = edge_props[e_in]
                # Score: prefer edges that share sum or product with many existing edges
                coll = 0
                for e in elist:
                    if e == e_out:
                        continue
                    s_e, p_e = edge_props[e]
                    if s_in == s_e:
                        coll += 1
                    if p_in == p_e:
                        coll += 1
                if coll > 0:
                    candidates.append((coll, e_in))
            
            if not candidates:
                continue
            
            # Pick the best candidate
            candidates.sort(reverse=True)
            e_in = candidates[0][1]
            
            new_score = score_delta(vals, elist, e_out, e_in)
            if new_score < cur_score:
                elist[i] = e_in
                edge_set.discard(e_out)
                edge_set.add(e_in)
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_elist = list(elist)
                    best_edge_set = set(edge_set)
        
        return best_score, best_edge_set

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
    # More random starts with small values
    for _ in range(10):
        vals = random.sample(range(1, 50), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))
    # Starts with multiplicative structure
    for _ in range(5):
        vals = multiplicative_values()
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    remaining_time = deadline - time.time() - 10.0
    per = max(2000, int(remaining_time / len(starts) / 0.00005))
    for vals0, edges0 in starts:
        if time.time() > deadline - 10.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges

    # Phase 3: targeted collision search
    if time.time() < deadline - 5.0:
        sc, edges = targeted_collision_search(best_vals, best_edges, budget=20000)
        if sc < best_score:
            best_score = sc
            best_edges = edges

    # Phase 4: polish edges for best values
    if time.time() < deadline - 3.0:
        elist = list(best_edges)
        it = 0
        while it < 30000 and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            s = score_delta(best_vals, elist, e_out, e_in)
            if s < best_score:
                elist[i] = e_in
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s

    # Phase 5: polish values for best edges
    if time.time() < deadline - 1.0:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 20000 and time.time() < deadline - 0.5:
            it += 1
            idx = random.randint(0, n - 1)
            r = random.random()
            if r < 0.4:
                new_val = random.randint(1, 60)
            elif r < 0.7:
                a = random.randint(1, 10)
                b = random.randint(1, 10)
                new_val = a * b
            else:
                new_val = vals[idx] + random.randint(-8, 8)
            
            if not (1 <= new_val <= 10000) or new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = score_of(new_vals, elist)
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
