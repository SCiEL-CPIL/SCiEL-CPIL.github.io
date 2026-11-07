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
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        u_out, v_out = e_out
        s_out = values[u_out] + values[v_out]
        p_out = values[u_out] * values[v_out]
        sum_counts[s_out] -= 1
        prod_counts[p_out] -= 1
        if sum_counts[s_out] == 0:
            del sum_counts[s_out]
        if prod_counts[p_out] == 0:
            del prod_counts[p_out]
        u_in, v_in = e_in
        s_in = values[u_in] + values[v_in]
        p_in = values[u_in] * values[v_in]
        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
        return max(len(sum_counts), len(prod_counts))

    def incremental_value_score(values, edge_list, idx, new_val):
        """Compute score after changing values[idx] to new_val, incrementally."""
        # Collect edges incident to idx
        incident = []
        non_incident_sums = set()
        non_incident_prods = set()
        for u, v in edge_list:
            if u == idx or v == idx:
                incident.append((u, v))
            else:
                non_incident_sums.add(values[u] + values[v])
                non_incident_prods.add(values[u] * values[v])
        
        new_sums = set(non_incident_sums)
        new_prods = set(non_incident_prods)
        for u, v in incident:
            other = v if u == idx else u
            new_sums.add(new_val + values[other])
            new_prods.add(new_val * values[other])
        
        return max(len(new_sums), len(new_prods))

    def try_target_k(vals, K, time_limit=2.0):
        """Try to find 35 edges with at most K distinct sums and K distinct products."""
        t_end = time.time() + time_limit
        edge_sums = {}
        edge_prods = {}
        for e in all_edges:
            s = vals[e[0]] + vals[e[1]]
            p = vals[e[0]] * vals[e[1]]
            edge_sums[e] = s
            edge_prods[e] = p
        
        # Greedy: pick edges that maximize collision
        chosen = []
        chosen_set = set()
        active_sums = set()
        active_prods = set()
        
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
                sc = 0
                if s in active_sums:
                    sc += 10
                if p in active_prods:
                    sc += 10
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
        
        if len(active_sums) <= K and len(active_prods) <= K:
            return chosen
        
        # Local search
        cur = list(chosen)
        cur_max = max(len(active_sums), len(active_prods))
        
        it = 0
        while time.time() < t_end and it < 5000:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            new_cur = list(cur)
            new_cur[i] = e_in
            new_sums = set(edge_sums[e] for e in new_cur)
            new_prods = set(edge_prods[e] for e in new_cur)
            new_max = max(len(new_sums), len(new_prods))
            if new_max <= cur_max:
                cur = new_cur
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
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
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 80)
                if new_val in vals:
                    continue
                s = incremental_value_score(vals, elist, idx, new_val)
                if s <= cur_score:
                    vals[idx] = new_val
                    cur_score = s
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
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
    ]
    for pool in pools:
        if time.time() > deadline - 40:
            break
        for _ in range(5):
            vals = random.sample(pool, n)
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

    # Phase 4: Polish values for best edges with incremental scoring
    if best_vals is not None and time.time() < deadline - 1.0:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 20000 and time.time() < deadline - 0.5:
            it += 1
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, 100)
            if new_val in vals:
                continue
            s = incremental_value_score(vals, elist, idx, new_val)
            if s < best_score:
                vals[idx] = new_val
                best_score = s
                best_vals = list(vals)

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
