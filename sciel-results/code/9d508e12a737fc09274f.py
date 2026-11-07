import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    n_all_edges = len(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score(values, edge_list):
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))

    # Precompute for each vertex the list of edges incident to it
    vertex_edges = [[] for _ in range(n)]
    for e in all_edges:
        vertex_edges[e[0]].append(e)
        vertex_edges[e[1]].append(e)

    def optimize_values_for_edges(vals, edges, budget_iters=50000):
        """
        Fine-tune values for fixed edges.
        """
        vals = list(vals)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        
        edge_arr = [(e[0], e[1]) for e in edges]
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            r = random.random()
            new_val = None
            if r < 0.30:
                new_val = old_val + random.randint(-3, 3)
            elif r < 0.50:
                new_val = old_val + random.randint(-15, 15)
            elif r < 0.65:
                new_val = random.randint(1, max(2, old_val * 2))
            elif r < 0.85:
                # Try to match an existing sum
                target = random.choice([vals[u] + vals[v] for u, v in edge_arr])
                for e in vertex_edges[idx]:
                    other = e[0] if e[1] == idx else e[1]
                    cand = target - vals[other]
                    if 1 <= cand <= 10000 and cand != old_val:
                        if not any(v == cand for j, v in enumerate(vals) if j != idx):
                            new_val = cand
                            break
                if new_val is None:
                    continue
            else:
                # Try to match a product
                target = random.choice([vals[u] * vals[v] for u, v in edge_arr])
                for e in vertex_edges[idx]:
                    other = e[0] if e[1] == idx else e[1]
                    if vals[other] != 0 and target % vals[other] == 0:
                        cand = target // vals[other]
                        if 1 <= cand <= 10000 and cand != old_val:
                            if not any(v == cand for j, v in enumerate(vals) if j != idx):
                                new_val = cand
                                break
                if new_val is None:
                    continue
            
            if new_val is None or new_val <= 0 or new_val > 10000:
                continue
            if any(v == new_val for j, v in enumerate(vals) if j != idx):
                continue
            
            new_vals = list(vals)
            new_vals[idx] = new_val
            new_score = score_of(new_vals, edges)
            
            if new_score < cur_score:
                vals = new_vals
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(vals)
            elif new_score == cur_score and random.random() < 0.3:
                vals = new_vals
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 3000 and random.random() < 0.01:
                no_improve = 0

        return best_score, best_vals

    def optimize_edges_for_values(vals, edges, budget_iters=50000):
        """
        Optimize edges for fixed values.
        """
        edges = list(edges)
        edges_set = set(edges)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_edges = list(edges)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            move_type = random.random()
            
            if move_type < 0.7:
                # Single edge swap
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(vals, new_edges)
                
                if new_score < cur_score:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.2:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Two-edge swap
                i1 = random.randint(0, m - 1)
                i2 = random.randint(0, m - 1)
                if i1 == i2:
                    continue
                e_out1 = edges[i1]
                e_out2 = edges[i2]
                e_in1 = random.choice(all_edges)
                e_in2 = random.choice(all_edges)
                if e_in1 in edges_set or e_in2 in edges_set or e_in1 == e_in2:
                    continue
                
                new_edges = list(edges)
                new_edges[i1] = e_in1
                new_edges[i2] = e_in2
                new_score = score_of(vals, new_edges)
                
                if new_score < cur_score:
                    edges = new_edges
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.15:
                    edges = new_edges
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    no_improve = 0
                else:
                    no_improve += 1
            
            if no_improve > 2000 and random.random() < 0.02:
                no_improve = 0

        return best_score, best_edges

    def joint_search(vals, edges, budget_iters=80000):
        """
        Joint local search on values and edges with smart moves.
        """
        vals = list(vals)
        edges = list(edges)
        edges_set = set(edges)
        
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        no_improve = 0
        it = 0
        
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            
            if random.random() < 0.55:
                # Edge move
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(vals, new_edges)
                
                if new_score < cur_score:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.25:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Value move
                idx = random.randint(0, n - 1)
                old_val = vals[idx]
                r = random.random()
                new_val = None
                if r < 0.4:
                    new_val = old_val + random.randint(-3, 3)
                elif r < 0.6:
                    new_val = old_val + random.randint(-12, 12)
                elif r < 0.8:
                    new_val = random.randint(1, max(2, old_val * 3))
                else:
                    # Targeted: try to create a collision
                    e_target = random.choice(edges)
                    target_sum = vals[e_target[0]] + vals[e_target[1]]
                    for e in vertex_edges[idx]:
                        other = e[0] if e[1] == idx else e[0]
                        cand = target_sum - vals[other]
                        if 1 <= cand <= 10000 and cand != old_val:
                            if not any(v == cand for j, v in enumerate(vals) if j != idx):
                                new_val = cand
                                break
                    if new_val is None:
                        continue
                
                if new_val is None or new_val <= 0 or new_val > 10000:
                    continue
                if any(v == new_val for j, v in enumerate(vals) if j != idx):
                    continue
                
                new_vals = list(vals)
                new_vals[idx] = new_val
                new_score = score_of(new_vals, edges)
                
                if new_score < cur_score:
                    vals = new_vals
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.25:
                    vals = new_vals
                    no_improve = 0
                else:
                    no_improve += 1
            
            if no_improve > 4000 and random.random() < 0.008:
                no_improve = 0

        return best_score, best_vals, best_edges

    def construct_partition_solution(partition, vals):
        """
        Given a partition of vertices into groups, construct edges such that
        intra-group edges are included first, then fill with inter-group edges.
        """
        edges = []
        edge_set = set()
        
        for group in partition:
            for u, v in combinations(group, 2):
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    edges.append(e)
                    edge_set.add(e)
        
        remaining = m - len(edges)
        if remaining > 0:
            inter_edges = []
            for e in all_edges:
                if e not in edge_set:
                    inter_edges.append(e)
            
            inter_edges.sort(key=lambda e: (vals[e[0]] + vals[e[1]], vals[e[0]] * vals[e[1]]))
            
            for e in inter_edges:
                if remaining <= 0:
                    break
                edges.append(e)
                edge_set.add(e)
                remaining -= 1
        
        return edges[:m]

    def greedy_best_edges_for_values(values, budget):
        """Greedy + local search: pick 35 edges to minimize score, for fixed values."""
        chosen = []
        chosen_set = set()
        edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        for e in edge_order:
            if len(chosen) == m:
                break
            chosen.append(e)
            chosen_set.add(e)
        
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

    # Strategy: Try various partition structures and value assignments
    best_score = float('inf')
    best_vals = None
    best_edges = None
    
    random.seed(42)
    
    # Candidate partitions and value strategies
    candidates = []
    
    # Strategy 1: Groups of 4, 4, 3, 3 with consecutive values
    part1 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    vals1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidates.append((part1, vals1))
    
    # Strategy 2: Groups of 5, 4, 3, 2
    part2 = [[0,1,2,3,4], [5,6,7,8], [9,10,11], [12,13]]
    vals2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidates.append((part2, vals2))
    
    # Strategy 3: One big group of 6, rest scattered
    part3 = [[0,1,2,3,4,5], [6,7,8,9], [10,11,12], [13]]
    vals3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidates.append((part3, vals3))
    
    # Strategy 4: Values with more structure - use small values to get product collisions
    vals4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    part4 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part4, vals4))
    
    # Strategy 5: Powers and small values
    vals5 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    part5 = [[0,1,2,3,4], [5,6,7,8], [9,10,11], [12,13]]
    candidates.append((part5, vals5))
    
    # Strategy 6: Try to make products collide - factor-rich values
    vals6 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    part6 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part6, vals6))
    
    # Strategy 7: Arithmetic with small difference
    vals7 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    part7 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part7, vals7))
    
    # Strategy 8: Fibonacci-like
    vals8 = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    part8 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part8, vals8))
    
    # Strategy 9: Clustered small values
    vals9 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    part9 = [[0,1,2,3,4,5], [6,7,8,9,10], [11,12,13]]
    candidates.append((part9, vals9))
    
    # Strategy 10: Random small values
    for _ in range(5):
        rv = random.sample(range(1, 30), n)
        part = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
        candidates.append((part, rv))
    
    # For each candidate, construct edges and optimize
    for part, vals in candidates:
        if time.time() > deadline - 20.0:
            break
        
        vals = list(vals)
        edges = construct_partition_solution(part, vals)
        
        if len(edges) != m:
            continue
        
        # Optimize edges for these values
        sc, edges_opt = optimize_edges_for_values(vals, edges, budget_iters=12000)
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = edges_opt
        
        # Also try greedy edge selection
        if time.time() < deadline - 15.0:
            edges_g, sc_g = greedy_best_edges_for_values(vals, budget=8000)
            if sc_g < best_score:
                best_score = sc_g
                best_vals = list(vals)
                best_edges = edges_g
            # Continue from greedy with joint
            if time.time() < deadline - 12.0:
                sc_j, vals_j, edges_j = joint_search(vals, edges_g, budget_iters=15000)
                if sc_j < best_score:
                    best_score = sc_j
                    best_vals = vals_j
                    best_edges = edges_j
        
        # Optimize values for these edges
        if time.time() > deadline - 10.0:
            break
        sc, vals_opt = optimize_values_for_edges(vals, edges_opt, budget_iters=12000)
        if sc < best_score:
            best_score = sc
            best_vals = vals_opt
            best_edges = edges_opt
        
        # Joint search
        if time.time() > deadline - 8.0:
            break
        sc, vals_j, edges_j = joint_search(vals_opt, edges_opt, budget_iters=20000)
        if sc < best_score:
            best_score = sc
            best_vals = vals_j
            best_edges = edges_j
    
    # Additional: pure random restarts with joint search
    for _ in range(6):
        if time.time() > deadline - 10.0:
            break
        rv = random.sample(range(1, 60), n)
        re = random.sample(all_edges, m)
        sc, vals_j, edges_j = joint_search(rv, re, budget_iters=12000)
        if sc < best_score:
            best_score = sc
            best_vals = vals_j
            best_edges = edges_j
    
    # Final polish
    if best_vals is not None and time.time() < deadline - 5.0:
        sc, vals_f = optimize_values_for_edges(best_vals, best_edges, budget_iters=15000)
        if sc < best_score:
            best_score = sc
            best_vals = vals_f
        
        sc, edges_f = optimize_edges_for_values(best_vals, best_edges, budget_iters=15000)
        if sc < best_score:
            best_score = sc
            best_edges = edges_f
        
        # One more joint pass
        if time.time() < deadline - 3.0:
            sc, vals_f2, edges_f2 = joint_search(best_vals, best_edges, budget_iters=15000)
            if sc < best_score:
                best_score = sc
                best_vals = vals_f2
                best_edges = edges_f2
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    # Validate and convert
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    assert len(final_values) == 14
    assert len(set(final_values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == 35
    edge_set_check = set()
    for e in final_edges:
        assert isinstance(e, list) and len(e) == 2
        assert isinstance(e[0], int) and isinstance(e[1], int)
        assert 0 <= e[0] < 14 and 0 <= e[1] < 14
        assert e[0] != e[1]
        key = tuple(sorted(e))
        assert key not in edge_set_check
        edge_set_check.add(key)
    
    return {'values': final_values, 'edges': final_edges}
