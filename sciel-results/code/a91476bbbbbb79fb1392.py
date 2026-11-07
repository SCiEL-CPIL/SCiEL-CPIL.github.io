"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
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
    
    def evaluate_fast(values, edges, sums_set, prods_set):
        """Incremental evaluation: only recompute changed edges."""
        for u, v in edges:
            sums_set.add(values[u] + values[v])
            prods_set.add(values[u] * values[v])
        return max(len(sums_set), len(prods_set))
    
    def greedy_edges(values, all_edges, rng):
        """Pick edges greedily to maximize reuse of existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_tiebreak = float('inf')
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = 0
                if s not in sums_seen:
                    score += 1
                if p not in prods_seen:
                    score += 1
                # Tie breaker: prefer edges whose sum/product are closer to existing values
                if score < best_score or (score == best_score and (s + p) < best_tiebreak):
                    best_score = score
                    best_tiebreak = s + p
                    best_edge = edge
            if best_edge is None:
                break
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return selected
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges: swap one edge for another."""
        edges_set = set(tuple(e) for e in edges)
        current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try random swaps
            for _ in range(200):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
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
    
    def local_search_values(values, edges, rng, time_limit):
        """Local search on values: perturb one value at a time."""
        current_score = evaluate(values, edges)
        value_set = set(values)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(80):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                # Generate diverse candidates
                candidates = set()
                # Nearby values
                for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50, -10, -20, -50]:
                    candidates.add(old_val + delta)
                # Multiples of small numbers
                for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                    for mult in range(1, 20):
                        candidates.add(base * mult)
                # Random
                candidates.add(rng.randint(1, max_val))
                candidates.add(rng.randint(1, 500))
                # Highly composite-like numbers
                for hc in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 144, 168, 180, 240, 360, 720]:
                    candidates.add(hc)
                    candidates.add(hc * 2)
                    candidates.add(hc * 3)
                
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
    
    def exhaustive_polish(values, edges, time_limit):
        """Try all single-value changes and single-edge swaps to find improvements."""
        current_score = evaluate(values, edges)
        value_set = set(values)
        edges_set = set(tuple(e) for e in edges)
        
        # Try all single edge swaps
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(edges)):
                if (time.time() - start_time) > time_limit:
                    return values, current_score
                old_edge = edges[i]
                for new_edge in all_edges:
                    if tuple(new_edge) in edges_set:
                        continue
                    edges[i] = new_edge
                    edges_set.discard(tuple(old_edge))
                    edges_set.add(tuple(new_edge))
                    new_score = evaluate(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        edges[i] = old_edge
                        edges_set.discard(tuple(new_edge))
                        edges_set.add(tuple(old_edge))
        
        # Try all single value changes (sampled)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for idx in range(n):
                if (time.time() - start_time) > time_limit:
                    return values, current_score
                old_val = values[idx]
                # Try a range of candidate values
                candidates = set()
                for delta in range(-20, 21):
                    candidates.add(old_val + delta)
                for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 48, 60]:
                    for mult in range(1, 30):
                        candidates.add(base * mult)
                candidates.add(rng.randint(1, max_val))
                
                found = False
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
                        found = True
                        break
                    else:
                        values[idx] = old_val
                        value_set.discard(new_val)
                        value_set.add(old_val)
                if not found:
                    values[idx] = old_val
                    value_set.add(old_val)
        
        return values, current_score
    
    # Main search loop
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Better initial value strategies using highly composite numbers
    value_strategies = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54],
        [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24],
    ]
    
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 10.0:
            break
        
        init_values = list(init_values)
        
        # 1. Greedy edges
        edges = greedy_edges(init_values, all_edges, rng)
        
        # 2. Local search edges
        edges, score = local_search_edges(init_values, edges, rng, deadline - 15.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': list(edges)}
        
        # 3. Local search values
        values = list(init_values)
        values, score2 = local_search_values(values, edges, rng, deadline - 15.0)
        
        if score2 < best_overall_score:
            best_overall_score = score2
            best_result = {'values': list(values), 'edges': list(edges)}
            
        # 4. Refine edges with new values
        if (time.time() - start_time) < deadline - 15.0:
            edges3, score3 = local_search_edges(values, edges, rng, deadline - 15.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': list(edges3)}

    # Final exhaustive polish
    if best_result is not None and (time.time() - start_time) < deadline - 5.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        # Alternate between value and edge local search
        for _ in range(5):
            if (time.time() - start_time) > deadline - 5.0:
                break
            values, s1 = local_search_values(values, edges, rng, deadline - 5.0)
            edges, s2 = local_search_edges(values, edges, rng, deadline - 5.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': list(edges)}
        
        # Exhaustive polish
        values, s_final = exhaustive_polish(values, edges, deadline - 2.0)
        if s_final < best_overall_score:
            best_overall_score = s_final
            best_result = {'values': list(values), 'edges': list(edges)}

    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
