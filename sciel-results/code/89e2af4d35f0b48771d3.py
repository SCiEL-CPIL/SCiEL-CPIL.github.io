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

    def value_move_score(values, idx, new_val, edge_list):
        """Compute new score after changing values[idx] to new_val."""
        sums = set()
        prods = set()
        for u, v in edge_list:
            vu = values[u] if u != idx else new_val
            vv = values[v] if v != idx else new_val
            sums.add(vu + vv)
            prods.add(vu * vv)
        return max(len(sums), len(prods))

    def find_good_edges_for_values(vals, time_limit=5.0):
        """Find edges that minimize distinct sums and products for given values."""
        t_end = time.time() + time_limit
        
        # Precompute sums and products for all edges
        edge_sums = {}
        edge_prods = {}
        for e in all_edges:
            edge_sums[e] = vals[e[0]] + vals[e[1]]
            edge_prods[e] = vals[e[0]] * vals[e[1]]
        
        # Group edges by sum and product
        sum_to_edges = {}
        prod_to_edges = {}
        for e in all_edges:
            s = edge_sums[e]
            p = edge_prods[e]
            if s not in sum_to_edges:
                sum_to_edges[s] = []
            sum_to_edges[s].append(e)
            if p not in prod_to_edges:
                prod_to_edges[p] = []
            prod_to_edges[p].append(e)
        
        best_edges = None
        best_score = float('inf')
        
        # Multiple random restarts
        for restart in range(30):
            if time.time() > t_end:
                break
            
            # Start with random edges
            chosen = set(random.sample(all_edges, min(10, m)))
            active_sums = set(edge_sums[e] for e in chosen)
            active_prods = set(edge_prods[e] for e in chosen)
            
            # Greedily add edges
            while len(chosen) < m:
                if time.time() > t_end:
                    break
                best_e = None
                best_sc = -float('inf')
                
                for e in all_edges:
                    if e in chosen:
                        continue
                    s = edge_sums[e]
                    p = edge_prods[e]
                    sc = 0
                    if s in active_sums:
                        sc += 10
                    if p in active_prods:
                        sc += 10
                    sc += len(sum_to_edges.get(s, [])) * 0.3
                    sc += len(prod_to_edges.get(p, [])) * 0.3
                    if sc > best_sc:
                        best_sc = sc
                        best_e = e
                
                if best_e is None:
                    break
                chosen.add(best_e)
                active_sums.add(edge_sums[best_e])
                active_prods.add(edge_prods[best_e])
            
            if len(chosen) < m:
                continue
            
            cur_edges = list(chosen)
            cur_score = score_of(vals, cur_edges)
            
            if cur_score < best_score:
                best_score = cur_score
                best_edges = cur_edges
            
            # Local search
            cur = cur_edges
            cur_set = set(cur)
            it = 0
            while it < 5000 and time.time() < t_end:
                it += 1
                i = random.randint(0, m - 1)
                e_out = cur[i]
                e_in = random.choice(all_edges)
                if e_in in cur_set:
                    continue
                
                new_score = fast_score_delta(vals, cur, e_out, e_in)
                if new_score <= cur_score:
                    cur[i] = e_in
                    cur_set.discard(e_out)
                    cur_set.add(e_in)
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = list(cur)
        
        return best_score, best_edges

    def find_good_values_for_edges(edges, time_limit=5.0):
        """Find values that minimize distinct sums and products for given edges."""
        t_end = time.time() + time_limit
        best_vals = None
        best_score = float('inf')
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > t_end:
                break
            
            # Random initial values
            vals = random.sample(range(1, 100), n)
            cur_score = score_of(vals, edges)
            
            # Hill climbing with multiple value changes
            improved = True
            it = 0
            while improved and it < 1000 and time.time() < t_end:
                improved = False
                it += 1
                for idx in range(n):
                    # Try nearby values
                    for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -10, 10, -20, 20]:
                        new_val = vals[idx] + delta
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in vals and vals.index(new_val) != idx:
                            continue
                        new_vals = list(vals)
                        new_vals[idx] = new_val
                        s = value_move_score(vals, idx, new_val, edges)
                        if s < cur_score:
                            vals = new_vals
                            cur_score = s
                            improved = True
                            break
            
            # Random jumps
            for _ in range(100):
                if time.time() > t_end:
                    break
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 200)
                if new_val in vals and vals.index(new_val) != idx:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = value_move_score(vals, idx, new_val, edges)
                if s < cur_score:
                    vals = new_vals
                    cur_score = s
            
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
        
        return best_score, best_vals

    def joint_search(vals0, edges0, budget):
        """Alternate value and edge moves."""
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
                # Edge move
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
                # Value move
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 150)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = value_move_score(vals, idx, new_val, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: Try different value pools and find good edges
    pools = [
        list(range(1, 30)),
        list(range(2, 40)),
        list(range(1, 25)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 21],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
        list(range(1, 50)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]
    
    for pool in pools:
        if time.time() > deadline - 40:
            break
        for _ in range(10):
            if time.time() > deadline - 40:
                break
            vals = random.sample(pool, n)
            sc, edges = find_good_edges_for_values(vals, time_limit=3.0)
            if sc < best_score:
                best_score = sc
                best_vals = list(vals)
                best_edges = set(edges)

    # Phase 2: Find good values for best edges
    if best_edges is not None and time.time() < deadline - 25:
        edges_list = list(best_edges)
        sc, vals = find_good_values_for_edges(edges_list, time_limit=15.0)
        if sc < best_score:
            best_score = sc
            best_vals = vals

    # Phase 3: Joint search from multiple starts
    random.seed(1234)
    starts = []
    if best_vals is not None and best_edges is not None:
        starts.append((list(best_vals), set(best_edges)))
    for _ in range(10):
        vals = random.sample(range(1, 80), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    remaining = deadline - time.time() - 10.0
    if remaining > 15:
        per = max(3000, int(remaining / len(starts) / 0.0001))
        for vals0, edges0 in starts:
            if time.time() > deadline - 10.0:
                break
            sc, vals, edges = joint_search(vals0, edges0, per)
            if sc < best_score:
                best_score = sc
                best_vals = vals
                best_edges = edges

    # Phase 4: Polish edges
    if best_vals is not None and time.time() < deadline - 5.0:
        elist = list(best_edges)
        it = 0
        while it < 30000 and time.time() < deadline - 2.0:
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

    # Phase 5: Polish values
    if best_vals is not None and time.time() < deadline - 1.0:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 20000 and time.time() < deadline - 0.5:
            it += 1
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, 200)
            if new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = value_move_score(vals, idx, new_val, elist)
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
