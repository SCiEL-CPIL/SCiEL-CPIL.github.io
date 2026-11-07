"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def count_distinct(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return len(sums), len(prods)
    
    def greedy_edges_optimized(values, all_edges, rng):
        """Greedy: pick edges that minimize the increase in distinct sums/products."""
        sums_seen = set()
        prods_seen = set()
        edges = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_s, best_p = 0, 0
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = (0 if s in sums_seen else 1) + (0 if p in prods_seen else 1)
                if score < best_score:
                    best_score = score
                    best_edge = e
                    best_s, best_p = s, p
                    if best_score == 0:
                        break
            if best_edge is None:
                best_edge = remaining[0]
                best_s = values[best_edge[0]] + values[best_edge[1]]
                best_p = values[best_edge[0]] * values[best_edge[1]]
            edges.append(best_edge)
            remaining.remove(best_edge)
            sums_seen.add(best_s)
            prods_seen.add(best_p)
        
        return edges

    def local_search(values, edges, rng, time_limit=2.0):
        """Intensive local search with edge swaps and value adjustments."""
        current_values = list(values)
        current_edges = [list(e) for e in edges]
        current_score = evaluate(current_values, current_edges)
        
        best_values = list(current_values)
        best_edges = [list(e) for e in current_edges]
        best_score = current_score
        
        all_edges = list(combinations(range(n), 2))
        edge_set = set(tuple(sorted(e)) for e in current_edges)
        
        start_t = time.time()
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_t < time_limit:
            if time.time() > deadline - 5:
                break
            
            if no_improve_count > max_no_improve:
                break
            
            move_type = rng.random()
            improved = False
            
            if move_type < 0.4:
                # Edge swap: replace one edge with another
                idx = rng.randrange(len(current_edges))
                old_edge = tuple(sorted(current_edges[idx]))
                
                # Find candidate non-edges
                non_edges = [e for e in all_edges if e not in edge_set]
                if not non_edges:
                    continue
                
                # Try multiple candidates
                candidates = rng.sample(non_edges, min(5, len(non_edges)))
                best_candidate = None
                best_candidate_score = current_score
                
                for new_edge in candidates:
                    new_edges = [list(e) for e in current_edges]
                    new_edges[idx] = list(new_edge)
                    new_score = evaluate(current_values, new_edges)
                    if new_score < best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = new_edge
                
                if best_candidate is not None:
                    current_edges[idx] = list(best_candidate)
                    edge_set.discard(old_edge)
                    edge_set.add(tuple(sorted(best_candidate)))
                    current_score = best_candidate_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [list(e) for e in current_edges]
                        best_values = list(current_values)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
            elif move_type < 0.7:
                # Value swap
                i, j = rng.sample(range(n), 2)
                new_values = list(current_values)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                
                if len(set(new_values)) != n:
                    continue
                
                new_score = evaluate(new_values, current_edges)
                if new_score <= current_score:
                    current_values = new_values
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = [list(e) for e in current_edges]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Value mutation: change one value to a nearby number
                i = rng.randrange(n)
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
                new_val = current_values[i] + delta
                if 1 <= new_val <= max_val:
                    new_values = list(current_values)
                    new_values[i] = new_val
                    if len(set(new_values)) == n:
                        new_score = evaluate(new_values, current_edges)
                        if new_score <= current_score:
                            current_values = new_values
                            current_score = new_score
                            improved = True
                            
                            if current_score < best_score:
                                best_score = current_score
                                best_values = list(current_values)
                                best_edges = [list(e) for e in current_edges]
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            if not improved:
                no_improve_count += 1
        
        return best_values, best_edges, best_score

    rng = random.Random(42)
    all_edges = list(combinations(range(n), 2))
    
    best_overall = float('inf')
    best_result = None
    
    # Generate value sets with strong structural properties
    value_sets = []
    
    # Strategy 1: Arithmetic progressions (minimize sum diversity)
    for start in range(1, 20):
        for step in range(1, 15):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                value_sets.append(vals)
    
    # Strategy 2: Values that are multiples of a common factor (minimize product diversity)
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for offset in range(0, 4):
            vals = [base * (i + 1) + offset for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                value_sets.append(vals)
    
    # Strategy 3: Small values from 1..30 (maximize collision probability)
    for _ in range(100):
        vals = rng.sample(range(1, 31), n)
        value_sets.append(vals)
    
    # Strategy 4: Values with many common factors
    # Use numbers like 6,12,18,24,... or 10,20,30,...
    for base in [6, 10, 12, 15, 20]:
        for start_mult in range(1, 5):
            vals = []
            for i in range(n):
                v = base * (start_mult + i)
                if v <= max_val:
                    vals.append(v)
            if len(vals) == n and len(set(vals)) == n:
                value_sets.append(vals)
    
    # Strategy 5: Mixed - some multiples, some not
    for _ in range(50):
        base = rng.choice([2, 3, 4, 5])
        vals = []
        for i in range(n):
            if rng.random() < 0.7:
                v = base * rng.randint(1, 20)
            else:
                v = rng.randint(1, 30)
            while v in vals or v < 1 or v > max_val:
                v = rng.randint(1, 50)
            vals.append(v)
        if len(set(vals)) == n:
            value_sets.append(vals)
    
    # Deduplicate
    seen_sets = set()
    unique_value_sets = []
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_value_sets.append(vs)
    
    # Sort by initial potential: prefer sets with more additive/multiplicative structure
    def structure_score(vs):
        """Lower is better: count how many sum/product collisions exist in complete graph"""
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i + 1, n):
                sums.add(vs[i] + vs[j])
                prods.add(vs[i] * vs[j])
        return max(len(sums), len(prods))
    
    unique_value_sets.sort(key=structure_score)
    
    # Run search for top value sets
    max_sets_to_try = 30
    sets_to_try = unique_value_sets[:max_sets_to_try]
    
    time_per_set = 60.0 / max_sets_to_try  # ~2 seconds per set
    
    for vs in sets_to_try:
        if time.time() > deadline - 10:
            break
        
        # Greedy edge initialization
        edges = greedy_edges_optimized(vs, all_edges, rng)
        
        # Local search refinement
        remaining_time = min(time_per_set, deadline - 5 - time.time())
        if remaining_time < 0.5:
            break
            
        vals, eds, score = local_search(vs, edges, rng, time_limit=remaining_time)
        
        if score < best_overall:
            best_overall = score
            best_result = (list(vals), [list(e) for e in eds])
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_edges[:m]]
    else:
        vals, edges = best_result
    
    # Final validation
    assert len(vals) == n
    assert len(set(vals)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in vals)
    assert len(edges) == m
    assert all(isinstance(u, int) and isinstance(v, int) and 0 <= u < n and 0 <= v < n and u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': vals, 'edges': edges}
# EVOLVE-BLOCK-END
