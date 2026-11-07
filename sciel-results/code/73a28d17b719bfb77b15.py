import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 84.0

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

    def fast_score_delta(values, edge_list, e_out, e_in):
        """Compute new score after removing e_out and adding e_in."""
        # Build current sums/prods with counts
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        # Remove e_out
        u_out, v_out = e_out
        s_out = values[u_out] + values[v_out]
        p_out = values[u_out] * values[v_out]
        sum_counts[s_out] -= 1
        prod_counts[p_out] -= 1
        if sum_counts[s_out] == 0:
            del sum_counts[s_out]
        if prod_counts[p_out] == 0:
            del prod_counts[p_out]
        # Add e_in
        u_in, v_in = e_in
        s_in = values[u_in] + values[v_in]
        p_in = values[u_in] * values[v_in]
        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
        return max(len(sum_counts), len(prod_counts))

    def try_target_k(vals, K, time_limit=2.0):
        """Try to find 35 edges with at most K distinct sums and K distinct products.
        Uses greedy construction driven by collision seeking."""
        t_end = time.time() + time_limit
        # Precompute all edge sums and products
        edge_sums = {}
        edge_prods = {}
        for e in all_edges:
            s = vals[e[0]] + vals[e[1]]
            p = vals[e[0]] * vals[e[1]]
            edge_sums[e] = s
            edge_prods[e] = p
        
        # Group edges by sum and by product
        sum_groups = {}
        prod_groups = {}
        for e in all_edges:
            s = edge_sums[e]
            p = edge_prods[e]
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append(e)
            if p not in prod_groups:
                prod_groups[p] = []
            prod_groups[p].append(e)
        
        # Greedy: pick edges that maximize collision
        chosen = []
        chosen_set = set()
        active_sums = set()
        active_prods = set()
        
        # Sort edges by how many collisions they would create
        for _ in range(m):
            if time.time() > t_end:
                return None
            best_e = None
            best_score = -1
            for e in all_edges:
                if e in chosen_set:
                    continue
                s = edge_sums[e]
                p = edge_prods[e]
                # Score: prefer edges whose sum or product is already active
                sc = 0
                if s in active_sums:
                    sc += 10
                if p in active_prods:
                    sc += 10
                # Bonus if both are new but close to existing
                if s not in active_sums and p not in active_prods:
                    sc -= 5
                if sc > best_score:
                    best_score = sc
                    best_e = e
            if best_e is None:
                return None
            chosen.append(best_e)
            chosen_set.add(best_e)
            active_sums.add(edge_sums[best_e])
            active_prods.add(edge_prods[best_e])
        
        # Check if we meet the target
        if len(active_sums) <= K and len(active_prods) <= K:
            return chosen
        
        # Local search to try to reduce
        cur = list(chosen)
        cur_sums = set(edge_sums[e] for e in cur)
        cur_prods = set(edge_prods[e] for e in cur)
        cur_max = max(len(cur_sums), len(cur_prods))
        
        it = 0
        while time.time() < t_end and it < 5000:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            # Compute new sets
            new_cur = list(cur)
            new_cur[i] = e_in
            new_sums = set(edge_sums[e] for e in new_cur)
            new_prods = set(edge_prods[e] for e in new_cur)
            new_max = max(len(new_sums), len(new_prods))
            if new_max <= cur_max:
                cur = new_cur
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                cur_sums = new_sums
                cur_prods = new_prods
                cur_max = new_max
                if cur_max <= K:
                    return cur
        return None

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
                # edge move
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                s = fast_score_delta(vals, elist, e_out, e_in)
                if s <= cur_score:
                    elist[i] = e_in
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score = s
            else:
                # value move
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 80)
                if new_val in vals:
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

    def targeted_value_move(vals, elist, cur_score):
        """Try value moves that create collisions with existing edge sums/products."""
        # Get current sums and products
        cur_sums = set()
        cur_prods = set()
        for u, v in elist:
            cur_sums.add(vals[u] + vals[v])
            cur_prods.add(vals[u] * vals[v])
        
        # For each vertex, try values that would create collisions
        for idx in range(n):
            # Find all edges incident to idx
            incident = [(u, v) for u, v in elist if u == idx or v == idx]
            if not incident:
                continue
            
            # Try values that make sum or product collide with existing
            candidates = set()
            for u, v in incident:
                other = v if u == idx else u
                # Sum collision: vals[idx] + vals[other] should be in cur_sums
                for s in cur_sums:
                    cand = s - vals[other]
                    if 1 <= cand <= 10000 and cand not in vals:
                        candidates.add(cand)
                # Product collision: vals[idx] * vals[other] should be in cur_prods
                for p in cur_prods:
                    if vals[other] != 0 and p % vals[other] == 0:
                        cand = p // vals[other]
                        if 1 <= cand <= 10000 and cand not in vals:
                            candidates.add(cand)
            
            # Try a few candidates
            for new_val in list(candidates)[:5]:
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, elist)
                if s < cur_score:
                    return new_vals, s
            # Also try a random small value
            new_val = random.randint(1, 30)
            if new_val not in vals:
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, elist)
                if s < cur_score:
                    return new_vals, s
        return None, cur_score

    def polish_edges_and_values(vals0, edges0, time_limit):
        """High-precision polish: alternate targeted value moves and edge moves."""
        vals = list(vals0)
        elist = list(edges0)
        cur_score = score_of(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(elist)
        t_end = time.time() + time_limit
        it = 0
        while time.time() < t_end and it < 50000:
            it += 1
            if random.random() < 0.4:
                # Targeted value move
                new_vals, s = targeted_value_move(vals, elist, cur_score)
                if new_vals is not None:
                    vals = new_vals
                    cur_score = s
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = set(elist)
            else:
                # Edge move
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in set(elist):
                    continue
                s = fast_score_delta(vals, elist, e_out, e_in)
                if s <= cur_score:
                    elist[i] = e_in
                    cur_score = s
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = set(elist)
        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: Target-K construction with various value pools
    pools = [
        list(range(1, 25)),
        list(range(2, 30)),
        list(range(1, 20)),
        [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
        [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
        [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
        list(range(1, 35)),
        [2,3,4,5,6,7,8,9,10,11,12,13,14,16],
        [1,2,3,4,5,6,8,9,10,12,15,18,20,24],
        [1,2,3,4,6,8,9,12,15,18,20,24,27,30],
    ]
    for pool in pools:
        if time.time() > deadline - 40:
            break
        for _ in range(5):
            vals = random.sample(pool, n)
            # Try multiple target K values
            for K in [10, 12, 14, 16, 18, 20]:
                result = try_target_k(vals, K, time_limit=1.5)
                if result is not None:
                    sc = score_of(vals, result)
                    if sc < best_score:
                        best_score = sc
                        best_vals = list(vals)
                        best_edges = set(result)
            if time.time() > deadline - 35:
                break

    # Phase 2: Joint search from multiple starts
    random.seed(1234)
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), set(best_edges)))
    for _ in range(6):
        vals = random.sample(range(1, 60), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    remaining = deadline - time.time() - 8.0
    if remaining > 10:
        per = max(1000, int(remaining / len(starts) / 0.0001))
        for vals0, edges0 in starts:
            if time.time() > deadline - 8.0:
                break
            sc, vals, edges = joint_search(vals0, edges0, per)
            if sc < best_score:
                best_score = sc
                best_vals = vals
                best_edges = edges

    # Phase 3: Polish edges for best values
    if best_vals is not None and time.time() < deadline - 4.0:
        elist = list(best_edges)
        it = 0
        while it < 15000 and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            s = fast_score_delta(best_vals, elist, e_out, e_in)
            if s < best_score:
                elist[i] = e_in
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s

    # Phase 4: High-precision polish with targeted value moves
    if best_vals is not None and time.time() < deadline - 1.0:
        polish_time = min(10.0, deadline - time.time() - 0.5)
        if polish_time > 1.0:
            sc, vals, edges = polish_edges_and_values(best_vals, best_edges, polish_time)
            if sc < best_score:
                best_score = sc
                best_vals = vals
                best_edges = edges

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
