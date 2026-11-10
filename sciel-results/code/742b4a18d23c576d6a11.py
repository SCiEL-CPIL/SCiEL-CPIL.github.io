"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_pairs = list(combinations(range(n), 2))
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def _greedy_construct(values, time_limit):
        """
        Greedily construct edges to minimize new distinct sums/products.
        """
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        edges = []
        edge_set = set()
        
        # Precompute sums and products for all pairs to speed up selection
        pair_props = {}
        for i, j in all_pairs:
            s = values[i] + values[j]
            p = values[i] * values[j]
            pair_props[(i, j)] = (s, p)
            
        # Candidate edges sorted by initial "cost" (not strictly, but we update dynamically)
        # We will just iterate through candidates repeatedly or use a priority queue logic manually.
        # Since m=35 is small, we can just pick the best available edge in each step.
        # O(m * |E|) = 35 * 91 = 3185 operations. Very fast.
        
        for _ in range(m):
            if time.time() > time_limit:
                break
            best_edge = None
            best_cost = float('inf')
            
            for i, j in all_pairs:
                if (i, j) in edge_set or (j, i) in edge_set:
                    continue
                s, p = pair_props[(i, j)]
                # Cost: 0 if both exist, 1 if one exists, 2 if none exist
                cost = 0
                if s not in sum_counts:
                    cost += 1
                if p not in prod_counts:
                    cost += 1
                
                # Tie breaker: prefer edges that increase counts of existing sums/prods more?
                # Or just random among ties to allow diversity in restarts.
                if cost < best_cost:
                    best_cost = cost
                    best_edge = (i, j)
                    # If cost is 0, we can break early? No, maybe another 0 is better for future?
                    # For now, take the first 0 found or randomize among 0s.
                    if cost == 0:
                        # Collect all 0-cost edges and pick one randomly
                        # This requires scanning all, so we don't break immediately
                        pass
            
            if best_edge is None:
                break # Should not happen as m < |E|
            
            i, j = best_edge
            s, p = pair_props[(i, j)]
            edges.append([i, j])
            edge_set.add((i, j))
            sum_counts[s] += 1
            prod_counts[p] += 1
            
        return edges, max(len(sum_counts), len(prod_counts))

    def _optimize_edges_fast(values, initial_edges, time_limit):
        best_edges = [list(e) for e in initial_edges]
        edge_set = set(map(tuple, best_edges))
        
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        def add_edge(edge):
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] += 1
            prod_counts[p] += 1
            edge_set.add(tuple(edge))
            best_edges.append(edge)
            
        def remove_edge(edge):
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] -= 1
            prod_counts[p] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            if prod_counts[p] == 0:
                del prod_counts[p]
            edge_set.remove(tuple(edge))
            best_edges.remove(edge)

        # Initialize
        for e in best_edges:
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] += 1
            prod_counts[p] += 1

        def get_score():
            return max(len(sum_counts), len(prod_counts))
        
        best_score = get_score()
        
        no_improve_count = 0
        while time.time() < time_limit and no_improve_count < 1000:
            # Pick random edge to remove
            idx = random.randint(0, len(best_edges) - 1)
            e_remove = best_edges[idx]
            
            # Pick random non-edge to add
            while True:
                i, j = random.choice(all_pairs)
                if (i, j) not in edge_set and (j, i) not in edge_set:
                    break
            
            e_add = [i, j]
            
            # Tentative swap
            remove_edge(e_remove)
            add_edge(e_add)
            
            new_score = get_score()
            
            if new_score < best_score:
                best_score = new_score
                no_improve_count = 0
            else:
                # Revert
                remove_edge(e_add)
                add_edge(e_remove)
                no_improve_count += 1
                
        return best_edges, best_score

    best_result = None
    best_overall_score = float('inf')
    
    # Generate candidate value sets
    # 1. Arithmetic Progressions (good for sum collisions)
    # 2. Smooth Numbers (good for product collisions)
    # 3. Mixed
    
    candidate_values = []
    
    # APs with small steps
    for start in range(1, 10):
        for step in [1, 2, 3]:
            vals = [start + i * step for i in range(n)]
            if max(vals) <= maximum and len(set(vals)) == n:
                candidate_values.append(vals)
                
    # Smooth numbers (5-smooth, 7-smooth)
    def is_smooth(num, max_prime):
        for p in [2, 3, 5, 7, 11]:
            if p > max_prime: break
            while num % p == 0:
                num //= p
        return num == 1

    smooth_5 = [i for i in range(1, 200) if is_smooth(i, 5)]
    if len(smooth_5) >= n:
        candidate_values.append(sorted(smooth_5[:n]))
        candidate_values.append(sorted(smooth_5[5:5+n]))
        
    smooth_7 = [i for i in range(1, 200) if is_smooth(i, 7)]
    if len(smooth_7) >= n:
        candidate_values.append(sorted(smooth_7[:n]))
        
    # Random compact sets
    random.seed(42)
    for _ in range(10):
        # Bias towards small numbers
        vals = random.sample(range(1, 40), n)
        candidate_values.append(sorted(vals))
        
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for v in candidate_values:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    candidate_values = unique_candidates

    # Run optimization
    # We have ~20-30 candidates.
    # Time per candidate: 
    # Greedy is fast (<0.1s).
    # Local search: 1-2s per restart.
    # Let's do 1 greedy + 2 local search restarts per candidate.
    
    time_per_candidate = (deadline - time.time()) / max(1, len(candidate_values))
    
    for vals in candidate_values:
        if len(set(vals)) != n or any(v <= 0 or v > maximum for v in vals):
            continue
            
        if time.time() > deadline - 10.0:
            break
            
        # 1. Greedy Start
        g_edges, g_score = _greedy_construct(vals, time.time() + 0.5)
        
        # 2. Local Search from Greedy
        ls_time_limit = time.time() + time_per_candidate * 0.4
        ls_edges, ls_score = _optimize_edges_fast(vals, g_edges, ls_time_limit)
        
        if ls_score < best_overall_score:
            best_overall_score = ls_score
            best_result = {'values': vals, 'edges': ls_edges}
            
        # 3. Local Search from Random
        if time.time() < deadline - 5.0:
            r_edges = [list(e) for e in random.sample(all_pairs, m)]
            ls2_time_limit = time.time() + time_per_candidate * 0.4
            ls2_edges, ls2_score = _optimize_edges_fast(vals, r_edges, ls2_time_limit)
            
            if ls2_score < best_overall_score:
                best_overall_score = ls2_score
                best_result = {'values': vals, 'edges': ls2_edges}
                
    if best_result is None:
        vals = list(range(1, n+1))
        edges = [list(e) for e in all_pairs[:m]]
        best_result = {'values': vals, 'edges': edges}
        
    result = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(i), int(j)] for i, j in best_result['edges']]
    }
    
    return result

# EVOLVE-BLOCK-END
