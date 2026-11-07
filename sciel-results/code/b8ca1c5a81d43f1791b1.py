"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def edge_sum_prod(values, u, v):
        return values[u] + values[v], values[u] * values[v]
    
    def build_edges_for_targets(values, target_sums, target_prods, rng):
        """Select m edges whose sums are in target_sums and products in target_prods.
        If not all m can be satisfied, pick the best m edges by score."""
        scored_edges = []
        for edge in all_edges:
            u, v = edge
            s, p = edge_sum_prod(values, u, v)
            score = 0
            if s in target_sums:
                score += 2
            if p in target_prods:
                score += 2
            scored_edges.append((score, s, p, edge))
        
        # Sort by score descending, then by sum+product for tie-breaking
        scored_edges.sort(key=lambda x: (-x[0], x[1] + x[2]))
        
        selected = []
        seen = set()
        for score, s, p, edge in scored_edges:
            if len(selected) >= m:
                break
            if edge not in seen:
                selected.append(edge)
                seen.add(edge)
        
        # If we don't have m edges, fill with remaining
        if len(selected) < m:
            for score, s, p, edge in scored_edges:
                if len(selected) >= m:
                    break
                if edge not in seen:
                    selected.append(edge)
                    seen.add(edge)
        
        return selected[:m]
    
    def optimize_values_for_targets(target_sums, target_prods, initial_values, rng, time_limit):
        """Optimize values so that many edges have sums in target_sums and products in target_prods."""
        values = list(initial_values)
        value_set = set(values)
        
        def count_good_edges(vals):
            count = 0
            for u, v in all_edges:
                s = vals[u] + vals[v]
                p = vals[u] * vals[v]
                if s in target_sums and p in target_prods:
                    count += 1
            return count
        
        current_count = count_good_edges(values)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(100):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                
                # Generate candidates based on target sums and products
                candidates = set()
                for other_idx in range(n):
                    if other_idx == idx:
                        continue
                    other_val = values[other_idx]
                    # Try to make sum hit a target
                    for ts in target_sums:
                        cand = ts - other_val
                        if 1 <= cand <= max_val and cand != old_val:
                            candidates.add(cand)
                    # Try to make product hit a target
                    for tp in target_prods:
                        if other_val != 0 and tp % other_val == 0:
                            cand = tp // other_val
                            if 1 <= cand <= max_val and cand != old_val:
                                candidates.add(cand)
                
                # Also try nearby values and random
                for delta in range(-10, 11):
                    candidates.add(old_val + delta)
                candidates.add(rng.randint(1, max_val))
                candidates.add(rng.randint(1, 500))
                
                found_improvement = False
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in value_set:
                        continue
                    values[idx] = new_val
                    value_set.discard(old_val)
                    value_set.add(new_val)
                    new_count = count_good_edges(values)
                    if new_count > current_count:
                        current_count = new_count
                        improved = True
                        found_improvement = True
                        break
                    else:
                        values[idx] = old_val
                        value_set.discard(new_val)
                        value_set.add(old_val)
                if not found_improvement:
                    values[idx] = old_val
                    value_set.add(old_val)
        
        return values, current_count
    
    def refine_edges_local_search(values, edges, rng, time_limit):
        """Local search on edges to minimize distinct sums and products."""
        edges_set = set(tuple(e) for e in edges)
        current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(500):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges) - 1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def refine_values_local_search(values, edges, rng, time_limit):
        """Local search on values to minimize distinct sums and products."""
        current_score = evaluate(values, edges)
        value_set = set(values)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(100):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                
                candidates = set()
                # Get neighbors in the edge set
                neighbor_vals = set()
                for u, v in edges:
                    if u == idx:
                        neighbor_vals.add(values[v])
                    elif v == idx:
                        neighbor_vals.add(values[u])
                
                # Try values that create sums/products already present
                existing_sums = set()
                existing_prods = set()
                for u, v in edges:
                    existing_sums.add(values[u] + values[v])
                    existing_prods.add(values[u] * values[v])
                
                for nv in neighbor_vals:
                    for es in existing_sums:
                        cand = es - nv
                        if 1 <= cand <= max_val:
                            candidates.add(cand)
                    for ep in existing_prods:
                        if nv != 0 and ep % nv == 0:
                            cand = ep // nv
                            if 1 <= cand <= max_val:
                                candidates.add(cand)
                
                for delta in range(-15, 16):
                    candidates.add(old_val + delta)
                candidates.add(rng.randint(1, max_val))
                candidates.add(rng.randint(1, 500))
                
                found_improvement = False
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in value_set:
                        continue
                    values[idx] = new_val
                    value_set.discard(old_val)
                    value_set.add(new_val)
                    new_score = evaluate(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        break
                    else:
                        values[idx] = old_val
                        value_set.discard(new_val)
                        value_set.add(old_val)
                if not found_improvement:
                    values[idx] = old_val
                    value_set.add(old_val)
        
        return values, current_score
    
    # Main search: try multiple target sum/product sets
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Generate candidate target sets from initial value configurations
    value_strategies = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30],
        [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24, 30],
        [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    ]
    
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 15.0:
            break
        
        init_values = list(init_values)
        
        # Compute sums and products for all edges
        edge_sums = {}
        edge_prods = {}
        for edge in all_edges:
            u, v = edge
            s = init_values[u] + init_values[v]
            p = init_values[u] * init_values[v]
            edge_sums[edge] = s
            edge_prods[edge] = p
        
        # Find most common sums and products
        from collections import Counter
        sum_counts = Counter(edge_sums.values())
        prod_counts = Counter(edge_prods.values())
        
        # Try different target set sizes
        for target_size in [10, 12, 15, 18, 20, 25]:
            if (time.time() - start_time) > deadline - 15.0:
                break
            
            # Top target sums
            top_sums = [s for s, c in sum_counts.most_common(target_size)]
            # Top target products
            top_prods = [p for p, c in prod_counts.most_common(target_size)]
            
            target_sums_set = set(top_sums)
            target_prods_set = set(top_prods)
            
            # Optimize values to hit these targets
            values, good_count = optimize_values_for_targets(
                target_sums_set, target_prods_set, init_values, rng, deadline - 20.0
            )
            
            if (time.time() - start_time) > deadline - 15.0:
                break
            
            # Build edges that hit targets
            edges = build_edges_for_targets(values, target_sums_set, target_prods_set, rng)
            
            # Evaluate
            score = evaluate(values, edges)
            
            if score < best_overall_score:
                best_overall_score = score
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
            
            # Refine edges
            if (time.time() - start_time) < deadline - 10.0:
                edges, score2 = refine_edges_local_search(values, edges, rng, deadline - 10.0)
                if score2 < best_overall_score:
                    best_overall_score = score2
                    best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
            
            # Refine values
            if (time.time() - start_time) < deadline - 10.0:
                values, score3 = refine_values_local_search(values, edges, rng, deadline - 10.0)
                if score3 < best_overall_score:
                    best_overall_score = score3
                    best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
            
            # Final edge refinement
            if (time.time() - start_time) < deadline - 5.0:
                edges, score4 = refine_edges_local_search(values, edges, rng, deadline - 5.0)
                if score4 < best_overall_score:
                    best_overall_score = score4
                    best_result = {'values': list(values), 'edges': [list(e) for e in edges]}

    # Final polish on best result
    if best_result is not None and (time.time() - start_time) < deadline - 3.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        # Alternate refinements
        for _ in range(3):
            if (time.time() - start_time) > deadline - 3.0:
                break
            values, s1 = refine_values_local_search(values, edges, rng, deadline - 3.0)
            edges, s2 = refine_edges_local_search(values, edges, rng, deadline - 3.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n + 1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
