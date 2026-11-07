import time
import random
import math
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, leave 5s buffer
    
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

    def score_with_counts(values, edge_list):
        """Return (score, sum_counts, prod_counts) for incremental updates."""
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        return max(len(sum_counts), len(prod_counts)), sum_counts, prod_counts

    def optimize_edges(values, initial_edges=None, budget_iters=30000):
        """
        Find 35 edges for fixed values minimizing score.
        Uses collision-aware greedy init then local search with incremental scoring.
        """
        if initial_edges:
            edges = list(initial_edges)
        else:
            # Collision-aware greedy: prefer edges whose sum/product already seen
            edges = []
            edge_set = set()
            sum_counts = {}
            prod_counts = {}
            
            # Precompute sum and product for all edges
            edge_sums = {}
            edge_prods = {}
            for e in all_edges:
                edge_sums[e] = values[e[0]] + values[e[1]]
                edge_prods[e] = values[e[0]] * values[e[1]]
            
            # Sort by (sum, product) to group similar edges
            sorted_edges = sorted(all_edges, key=lambda e: (edge_sums[e], edge_prods[e]))
            
            for e in sorted_edges:
                if len(edges) == m:
                    break
                edges.append(e)
                edge_set.add(e)
                s = edge_sums[e]
                p = edge_prods[e]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        
        edges_set = set(edges)
        cur_score, sum_counts, prod_counts = score_with_counts(values, edges)
        
        no_improve = 0
        it = 0
        total_iters = budget_iters
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            move_type = random.random()
            
            if move_type < 0.6:
                # Single edge swap - prefer edges that create collisions
                i = random.randint(0, m - 1)
                e_out = edges[i]
                
                # Score potential incoming edges by collision potential
                s_out = values[e_out[0]] + values[e_out[1]]
                p_out = values[e_out[0]] * values[e_out[1]]
                
                # Find candidate edges that would create collisions
                candidates = []
                for e in all_edges:
                    if e in edges_set:
                        continue
                    s_in = values[e[0]] + values[e[1]]
                    p_in = values[e[0]] * values[e[1]]
                    # Score: prefer edges where sum or product already exists
                    score = 0
                    if s_in in sum_counts:
                        score += sum_counts[s_in]
                    if p_in in prod_counts:
                        score += prod_counts[p_in]
                    # Penalize if removing e_out creates new unique values
                    if sum_counts[s_out] == 1:
                        score -= 1
                    if prod_counts[p_out] == 1:
                        score -= 1
                    candidates.append((score, e))
                
                if not candidates:
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue
                else:
                    # Weighted random selection favoring high collision scores
                    max_score = max(c[0] for c in candidates)
                    min_score = min(c[0] for c in candidates)
                    if max_score == min_score:
                        e_in = random.choice([c[1] for c in candidates])
                    else:
                        # Exponential weighting
                        weights = [math.exp((c[0] - min_score) / (max_score - min_score + 1)) for c in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cum = 0
                        e_in = candidates[0][1]
                        for w, c in zip(weights, candidates):
                            cum += w
                            if r <= cum:
                                e_in = c[1]
                                break
                
                if e_in in edges_set:
                    continue
                
                s_in = values[e_in[0]] + values[e_in[1]]
                p_in = values[e_in[0]] * values[e_in[1]]
                
                # Simulate removal
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                # Simulate addition
                new_sum_cnt = sum_counts.get(s_in, 0)
                new_prod_cnt = prod_counts.get(p_in, 0)
                sum_counts[s_in] = new_sum_cnt + 1
                prod_counts[p_in] = new_prod_cnt + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                if new_score <= cur_score:
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        del prod_counts[p_in]
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
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
                
                s_out1 = values[e_out1[0]] + values[e_out1[1]]
                p_out1 = values[e_out1[0]] * values[e_out1[1]]
                s_out2 = values[e_out2[0]] + values[e_out2[1]]
                p_out2 = values[e_out2[0]] * values[e_out2[1]]
                s_in1 = values[e_in1[0]] + values[e_in1[1]]
                p_in1 = values[e_in1[0]] * values[e_in1[1]]
                s_in2 = values[e_in2[0]] + values[e_in2[1]]
                p_in2 = values[e_in2[0]] * values[e_in2[1]]
                
                # Remove out edges
                sum_counts[s_out1] -= 1
                if sum_counts[s_out1] == 0: del sum_counts[s_out1]
                prod_counts[p_out1] -= 1
                if prod_counts[p_out1] == 0: del prod_counts[p_out1]
                sum_counts[s_out2] -= 1
                if sum_counts[s_out2] == 0: del sum_counts[s_out2]
                prod_counts[p_out2] -= 1
                if prod_counts[p_out2] == 0: del prod_counts[p_out2]
                
                # Add in edges
                sum_counts[s_in1] = sum_counts.get(s_in1, 0) + 1
                prod_counts[p_in1] = prod_counts.get(p_in1, 0) + 1
                sum_counts[s_in2] = sum_counts.get(s_in2, 0) + 1
                prod_counts[p_in2] = prod_counts.get(p_in2, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                if new_score <= cur_score:
                    edges[i1] = e_in1
                    edges[i2] = e_in2
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    cur_score = new_score
                    no_improve = 0
                else:
                    # Revert
                    sum_counts[s_in1] -= 1
                    if sum_counts[s_in1] == 0: del sum_counts[s_in1]
                    prod_counts[p_in1] -= 1
                    if prod_counts[p_in1] == 0: del prod_counts[p_in1]
                    sum_counts[s_in2] -= 1
                    if sum_counts[s_in2] == 0: del sum_counts[s_in2]
                    prod_counts[p_in2] -= 1
                    if prod_counts[p_in2] == 0: del prod_counts[p_in2]
                    sum_counts[s_out1] = sum_counts.get(s_out1, 0) + 1
                    prod_counts[p_out1] = prod_counts.get(p_out1, 0) + 1
                    sum_counts[s_out2] = sum_counts.get(s_out2, 0) + 1
                    prod_counts[p_out2] = prod_counts.get(p_out2, 0) + 1
                    no_improve += 1
            
            if no_improve > 2000 and random.random() < 0.02:
                no_improve = 0

        return edges, cur_score

    def optimize_values(values, edges, budget_iters=30000):
        """
        Optimize values for fixed edges using incremental scoring with collision targeting.
        """
        vals = list(values)
        cur_score, sum_counts, prod_counts = score_with_counts(vals, edges)
        
        # Precompute which edges touch each vertex
        vertex_edges = [[] for _ in range(n)]
        for u, v in edges:
            vertex_edges[u].append(v)
            vertex_edges[v].append(u)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            # Try various perturbation strategies targeting collisions
            r = random.random()
            if r < 0.20:
                # Small step
                new_val = old_val + random.randint(-5, 5)
            elif r < 0.40:
                # Target existing sums: find a neighbor and an existing sum
                # such that new_val = target_sum - vals[nbr]
                if vertex_edges[idx]:
                    nbr = random.choice(vertex_edges[idx])
                    existing_sums = list(sum_counts.keys())
                    if existing_sums:
                        target_sum = random.choice(existing_sums)
                        new_val = target_sum - vals[nbr]
                        if new_val <= 0 or new_val > 10000:
                            new_val = old_val + random.randint(-20, 20)
                    else:
                        new_val = old_val + random.randint(-20, 20)
                else:
                    new_val = old_val + random.randint(-20, 20)
            elif r < 0.55:
                # Target existing products: find a neighbor and an existing product
                # such that new_val = target_prod // vals[nbr]
                if vertex_edges[idx]:
                    nbr = random.choice(vertex_edges[idx])
                    existing_prods = list(prod_counts.keys())
                    if existing_prods:
                        target_prod = random.choice(existing_prods)
                        if vals[nbr] != 0 and target_prod % vals[nbr] == 0:
                            new_val = target_prod // vals[nbr]
                            if new_val <= 0 or new_val > 10000:
                                new_val = old_val + random.randint(-20, 20)
                        else:
                            new_val = old_val + random.randint(-20, 20)
                    else:
                        new_val = old_val + random.randint(-20, 20)
                else:
                    new_val = old_val + random.randint(-20, 20)
            elif r < 0.70:
                new_val = old_val + random.randint(-30, 30)
            elif r < 0.80:
                # Multiplicative perturbation for product collisions
                new_val = old_val * random.choice([2, 3, 5])
                if new_val > 10000:
                    new_val = old_val // random.choice([2, 3])
                    if new_val <= 0:
                        new_val = old_val + random.randint(-10, 10)
            else:
                new_val = random.randint(1, 300)
            
            if new_val <= 0 or new_val > 10000:
                continue
            if any(v == new_val for j, v in enumerate(vals) if j != idx):
                continue
            
            # Incremental score update: only edges touching idx change
            for nbr in vertex_edges[idx]:
                s_old = old_val + vals[nbr]
                p_old = old_val * vals[nbr]
                s_new = new_val + vals[nbr]
                p_new = new_val * vals[nbr]
                
                sum_counts[s_old] -= 1
                if sum_counts[s_old] == 0: del sum_counts[s_old]
                prod_counts[p_old] -= 1
                if prod_counts[p_old] == 0: del prod_counts[p_old]
                
                sum_counts[s_new] = sum_counts.get(s_new, 0) + 1
                prod_counts[p_new] = prod_counts.get(p_new, 0) + 1
            
            new_score = max(len(sum_counts), len(prod_counts))
            
            if new_score <= cur_score:
                vals[idx] = new_val
                cur_score = new_score
                no_improve = 0
            else:
                # Revert
                for nbr in vertex_edges[idx]:
                    s_old = old_val + vals[nbr]
                    p_old = old_val * vals[nbr]
                    s_new = new_val + vals[nbr]
                    p_new = new_val * vals[nbr]
                    
                    sum_counts[s_new] -= 1
                    if sum_counts[s_new] == 0: del sum_counts[s_new]
                    prod_counts[p_new] -= 1
                    if prod_counts[p_new] == 0: del prod_counts[p_new]
                    
                    sum_counts[s_old] = sum_counts.get(s_old, 0) + 1
                    prod_counts[p_old] = prod_counts.get(p_old, 0) + 1
                no_improve += 1
            
            if no_improve > 3000 and random.random() < 0.01:
                no_improve = 0

        return vals, cur_score

    def joint_search(init_vals, init_edges, budget_iters=50000):
        """
        Joint local search on values and edges with incremental scoring and simulated annealing.
        """
        vals = list(init_vals)
        edges = list(init_edges)
        edges_set = set(edges)
        
        cur_score, sum_counts, prod_counts = score_with_counts(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        # Precompute vertex-edge incidence
        vertex_edges = [[] for _ in range(n)]
        for u, v in edges:
            vertex_edges[u].append(v)
            vertex_edges[v].append(u)
        
        no_improve = 0
        it = 0
        total_iters = budget_iters
        # Simulated annealing temperature schedule
        initial_temp = 1.0
        final_temp = 0.01
        
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            # Annealing schedule: exponential cooling
            progress = it / total_iters
            temp = initial_temp * (final_temp / initial_temp) ** progress
            
            if random.random() < 0.55:
                # Edge move with incremental scoring
                i = random.randint(0, m - 1)
                e_out = edges[i]
                
                # Prefer edges that create collisions
                s_out = vals[e_out[0]] + vals[e_out[1]]
                p_out = vals[e_out[0]] * vals[e_out[1]]
                
                # Build candidate list with collision scores
                candidates = []
                for e in all_edges:
                    if e in edges_set:
                        continue
                    s_in = vals[e[0]] + vals[e[1]]
                    p_in = vals[e[0]] * vals[e[1]]
                    score = 0
                    if s_in in sum_counts:
                        score += sum_counts[s_in]
                    if p_in in prod_counts:
                        score += prod_counts[p_in]
                    if sum_counts[s_out] == 1:
                        score -= 1
                    if prod_counts[p_out] == 1:
                        score -= 1
                    candidates.append((score, e))
                
                if not candidates:
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue
                else:
                    max_score = max(c[0] for c in candidates)
                    min_score = min(c[0] for c in candidates)
                    if max_score == min_score:
                        e_in = random.choice([c[1] for c in candidates])
                    else:
                        weights = [math.exp((c[0] - min_score) / (max_score - min_score + 1) * temp) for c in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cum = 0
                        e_in = candidates[0][1]
                        for w, c in zip(weights, candidates):
                            cum += w
                            if r <= cum:
                                e_in = c[1]
                                break
                
                if e_in in edges_set:
                    continue
                
                s_in = vals[e_in[0]] + vals[e_in[1]]
                p_in = vals[e_in[0]] * vals[e_in[1]]
                
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0: del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0: del prod_counts[p_out]
                
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                # Simulated annealing acceptance
                delta = new_score - cur_score
                if delta <= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                    # Update vertex_edges
                    vertex_edges[e_out[0]].remove(e_out[1])
                    vertex_edges[e_out[1]].remove(e_out[0])
                    vertex_edges[e_in[0]].append(e_in[1])
                    vertex_edges[e_in[1]].append(e_in[0])
                else:
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0: del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0: del prod_counts[p_in]
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    no_improve += 1
            else:
                # Value move with incremental scoring
                idx = random.randint(0, n - 1)
                old_val = vals[idx]
                r = random.random()
                if r < 0.25:
                    # Target existing sums
                    if vertex_edges[idx]:
                        nbr = random.choice(vertex_edges[idx])
                        existing_sums = list(sum_counts.keys())
                        if existing_sums:
                            target = random.choice(existing_sums)
                            new_val = target - vals[nbr]
                            if new_val <= 0 or new_val > 10000:
                                new_val = old_val + random.randint(-10, 10)
                        else:
                            new_val = old_val + random.randint(-10, 10)
                    else:
                        new_val = old_val + random.randint(-10, 10)
                elif r < 0.40:
                    # Target existing products
                    if vertex_edges[idx]:
                        nbr = random.choice(vertex_edges[idx])
                        existing_prods = list(prod_counts.keys())
                        if existing_prods:
                            target = random.choice(existing_prods)
                            if vals[nbr] != 0 and target % vals[nbr] == 0:
                                new_val = target // vals[nbr]
                                if new_val <= 0 or new_val > 10000:
                                    new_val = old_val + random.randint(-10, 10)
                            else:
                                new_val = old_val + random.randint(-10, 10)
                        else:
                            new_val = old_val + random.randint(-10, 10)
                    else:
                        new_val = old_val + random.randint(-10, 10)
                elif r < 0.55:
                    new_val = old_val + random.randint(-5, 5)
                elif r < 0.70:
                    new_val = random.randint(1, 150)
                elif r < 0.85:
                    new_val = old_val + random.randint(-20, 20)
                else:
                    new_val = random.randint(1, 500)
                
                if new_val <= 0 or new_val > 10000:
                    continue
                if any(v == new_val for j, v in enumerate(vals) if j != idx):
                    continue
                
                # Update only edges touching idx
                for nbr in vertex_edges[idx]:
                    s_old = old_val + vals[nbr]
                    p_old = old_val * vals[nbr]
                    s_new = new_val + vals[nbr]
                    p_new = new_val * vals[nbr]
                    
                    sum_counts[s_old] -= 1
                    if sum_counts[s_old] == 0: del sum_counts[s_old]
                    prod_counts[p_old] -= 1
                    if prod_counts[p_old] == 0: del prod_counts[p_old]
                    
                    sum_counts[s_new] = sum_counts.get(s_new, 0) + 1
                    prod_counts[p_new] = prod_counts.get(p_new, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                # Simulated annealing acceptance
                delta = new_score - cur_score
                if delta <= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    vals[idx] = new_val
                    cur_score = new_score
                    no_improve = 0
                else:
                    for nbr in vertex_edges[idx]:
                        s_old = old_val + vals[nbr]
                        p_old = old_val * vals[nbr]
                        s_new = new_val + vals[nbr]
                        p_new = new_val * vals[nbr]
                        
                        sum_counts[s_new] -= 1
                        if sum_counts[s_new] == 0: del sum_counts[s_new]
                        prod_counts[p_new] -= 1
                        if prod_counts[p_new] == 0: del prod_counts[p_new]
                        
                        sum_counts[s_old] = sum_counts.get(s_old, 0) + 1
                        prod_counts[p_old] = prod_counts.get(p_old, 0) + 1
                    no_improve += 1
            
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = list(edges)
            
            if no_improve > 5000 and random.random() < 0.005:
                no_improve = 0

        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Generate diverse candidate value sets
    candidates = []
    
    # 1. Small consecutive
    candidates.append(list(range(1, 15)))
    
    # 2. Powers of 2 (good for product collisions with small set)
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 3. Divisors of highly composite number (many product collisions)
    candidates.append([1, 2, 3, 4, 6, 12, 5, 10, 15, 20, 30, 60, 7, 14])
    
    # 4. Arithmetic progression (good for sum collisions)
    candidates.append([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    
    # 5. Mixed: small values with some structure
    candidates.append([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    
    # 6. Random small values (multiple seeds for diversity)
    random.seed(42)
    for _ in range(4):
        rv = random.sample(range(1, 40), n)
        candidates.append(rv)
    
    # 7. Values designed for sum collisions: pairs that sum to same value
    # e.g., (1,9),(2,8),(3,7),(4,6) all sum to 10, plus extras
    candidates.append([1, 9, 2, 8, 3, 7, 4, 6, 5, 11, 12, 13, 14, 15])
    
    # 8. Values designed for product collisions
    # (1,12),(2,6),(3,4) all product 12; (1,20),(2,10),(4,5) product 20
    candidates.append([1, 12, 2, 6, 3, 4, 20, 10, 5, 7, 8, 9, 11, 13])
    
    # 9. Geometric with small ratio
    geo = [1]
    for i in range(1, 14):
        geo.append(int(geo[-1] * 1.3) + 1)
    # Ensure distinct
    seen = set()
    unique_geo = []
    for v in geo:
        if v not in seen and 1 <= v <= 10000:
            seen.add(v)
            unique_geo.append(v)
    while len(unique_geo) < n:
        unique_geo.append(len(unique_geo) + 100)
    candidates.append(unique_geo[:n])
    
    # 10. Fibonacci-like
    fib = [1, 2]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    candidates.append(fib[:n])
    
    # 11. Highly structured: values with many additive/multiplicative relationships
    # Small primes and their products
    candidates.append([2, 3, 5, 6, 7, 10, 11, 14, 15, 21, 22, 30, 33, 35])
    
    # 12. Values centered around a common sum
    # Many pairs sum to 20: (1,19),(2,18),(3,17),(4,16),(5,15),(6,14),(7,13)
    candidates.append([1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14, 7, 13])
    
    # 13. Values with common product
    # Many pairs multiply to 36: (1,36),(2,18),(3,12),(4,9),(6,6) - but need distinct
    candidates.append([1, 36, 2, 18, 3, 12, 4, 9, 5, 7, 8, 10, 11, 14])
    
    # 14. Compact range with structure
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 15. Values that create many sum collisions in a dense graph
    # Arithmetic with small step
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Sort candidates by initial greedy score to prioritize promising ones
    def greedy_score(cand):
        edges_c, score_c = optimize_edges(cand, budget_iters=3000)
        return score_c, edges_c
    
    scored_candidates = []
    for cand in candidates:
        if time.time() > deadline - 25.0:
            break
        sc, ec = greedy_score(cand)
        scored_candidates.append((sc, cand, ec))
    
    # Sort by score, process best first
    scored_candidates.sort(key=lambda x: x[0])
    
    # For each candidate (best first), run joint search
    for sc_init, cand, edges_c in scored_candidates:
        if time.time() > deadline - 12.0:
            break
        
        if sc_init < best_score:
            best_score = sc_init
            best_vals = list(cand)
            best_edges = edges_c
        
        # Joint search - give more budget to better initial solutions
        if sc_init <= best_score:
            joint_budget = 20000
        else:
            joint_budget = 10000
        
        sc, vals, edges = joint_search(cand, edges_c, budget_iters=joint_budget)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges
    
    # Final polish: edge optimization on best values
    if time.time() < deadline - 4.0 and best_vals is not None:
        edges_final, score_final = optimize_edges(best_vals, best_edges, budget_iters=15000)
        if score_final < best_score:
            best_score = score_final
            best_edges = edges_final
    
    # Value optimization on best edges
    if time.time() < deadline - 3.0 and best_vals is not None:
        vals_final, score_final = optimize_values(best_vals, best_edges, budget_iters=15000)
        if score_final < best_score:
            best_score = score_final
            best_vals = vals_final
    
    # One more joint polish if time allows
    if time.time() < deadline - 2.0 and best_vals is not None:
        sc, vals, edges = joint_search(best_vals, best_edges, budget_iters=8000)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges
    
    # Final edge-only polish
    if time.time() < deadline - 1.0 and best_vals is not None:
        edges_final, score_final = optimize_edges(best_vals, best_edges, budget_iters=5000)
        if score_final < best_score:
            best_score = score_final
            best_edges = edges_final
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    # Validate and convert
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Ensure validity
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
