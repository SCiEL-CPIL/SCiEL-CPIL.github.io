"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        """Compute max(distinct sums, distinct products) over edges."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edges_for_values(values, rng):
        """Greedy edge selection: prefer edges that reuse existing sums/products."""
        all_pairs = list(combinations(range(n), 2))
        rng.shuffle(all_pairs)
        
        sums_seen = set()
        prods_seen = set()
        selected = []
        
        for u, v in all_pairs:
            if len(selected) >= m:
                break
            s = values[u] + values[v]
            p = values[u] * values[v]
            # Cost: prefer edges that add fewer new distinct values
            cost = (1 if s not in sums_seen else 0) + (1 if p not in prods_seen else 0)
            selected.append((cost, u, v))
        
        # Sort by cost and take top m
        selected.sort(key=lambda x: x[0])
        edges = [[u, v] for _, u, v in selected[:m]]
        
        # Ensure we have exactly m edges
        if len(edges) < m:
            existing = set(tuple(sorted(e)) for e in edges)
            for u, v in all_pairs:
                if len(edges) >= m:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
        
        return edges
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges: swap or replace edges to reduce objective."""
        edges = [list(e) for e in edges]
        current_score = evaluate(values, edges)
        best_score = current_score
        best_edges = [list(e) for e in edges]
        
        all_pairs = list(combinations(range(n), 2))
        pair_set = set(tuple(sorted(e)) for e in edges)
        
        while time.time() < time_limit:
            improved = False
            # Try swapping one edge for another
            for i in range(len(edges)):
                if time.time() >= time_limit:
                    break
                u1, v1 = edges[i]
                pair1 = tuple(sorted((u1, v1)))
                
                # Try replacing edge i with a random other edge
                for _ in range(20):
                    u2, v2 = rng.choice(all_pairs)
                    pair2 = tuple(sorted((u2, v2)))
                    if pair2 == pair1:
                        continue
                    if pair2 in pair_set:
                        continue
                    
                    new_edges = [list(e) for e in edges]
                    new_edges[i] = [u2, v2]
                    new_score = evaluate(values, new_edges)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = new_edges
                        edges = new_edges
                        pair_set.discard(pair1)
                        pair_set.add(pair2)
                        improved = True
                        break
            if not improved and time.time() > time_limit - 5:
                break
        
        return best_edges, best_score
    
    def generate_structured_values(rng):
        """Generate structured value sets that encourage sum/product collisions."""
        strategy = rng.randint(0, 4)
        
        if strategy == 0:
            # Arithmetic progression
            start = rng.randint(1, 50)
            step = rng.randint(1, 10)
            vals = [start + i * step for i in range(n)]
            if max(vals) > max_val:
                step = max(1, (max_val - start) // (n - 1))
                vals = [start + i * step for i in range(n)]
            # Ensure distinct
            if len(set(vals)) < n:
                vals = list(range(1, n + 1))
            return vals
        
        elif strategy == 1:
            # Geometric-ish progression (powers)
            base = rng.randint(2, 5)
            vals = []
            for i in range(n):
                v = base ** i
                if v > max_val:
                    v = rng.randint(1, max_val)
                vals.append(v)
            # Ensure distinct
            seen = set()
            unique_vals = []
            for v in vals:
                if v not in seen and v <= max_val:
                    seen.add(v)
                    unique_vals.append(v)
            while len(unique_vals) < n:
                v = rng.randint(1, max_val)
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)
            return unique_vals
        
        elif strategy == 2:
            # Small numbers with some structure
            base_vals = list(range(1, n + 1))
            # Perturb some
            for _ in range(rng.randint(0, 5)):
                idx = rng.randint(0, n - 1)
                base_vals[idx] = rng.randint(1, 50)
            # Ensure distinct
            seen = set()
            unique_vals = []
            for v in base_vals:
                if v not in seen and v <= max_val:
                    seen.add(v)
                    unique_vals.append(v)
            while len(unique_vals) < n:
                v = rng.randint(1, 50)
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)
            return unique_vals
        
        elif strategy == 3:
            # Random small numbers
            vals = []
            seen = set()
            while len(vals) < n:
                v = rng.randint(1, 100)
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
            return vals
        
        else:
            # Random numbers up to max_val
            vals = []
            seen = set()
            while len(vals) < n:
                v = rng.randint(1, max_val)
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
            return vals
    
    def optimize_values_local(values, edges, rng, time_limit):
        """Local search on values: try small perturbations."""
        values = list(values)
        current_score = evaluate(values, edges)
        best_score = current_score
        best_values = list(values)
        
        while time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() >= time_limit:
                    break
                old_val = values[i]
                # Try replacing value i with a nearby value
                for delta in range(-10, 11):
                    new_val = old_val + delta
                    if new_val <= 0 or new_val > max_val:
                        continue
                    if new_val in values:
                        continue
                    values[i] = new_val
                    new_score = evaluate(values, edges)
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(values)
                        improved = True
                    else:
                        values[i] = old_val
                if improved:
                    values = best_values
                    break
            if not improved:
                break
        
        return best_values, best_score
    
    # Main search loop
    best_overall = None
    best_overall_score = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Try multiple value sets with greedy edges
    num_value_trials = 50
    for trial in range(num_value_trials):
        if time.time() >= deadline - 10:
            break
        
        values = generate_structured_values(rng)
        edges = greedy_edges_for_values(values, rng)
        score = evaluate(values, edges)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall = (list(values), [list(e) for e in edges])
    
    # Phase 2: Local search on edges for best value set
    if best_overall:
        values, edges = best_overall
        time_left = deadline - time.time()
        if time_left > 5:
            edges, score = local_search_edges(values, edges, rng, time.time() + min(time_left - 2, 30))
            if score < best_overall_score:
                best_overall_score = score
                best_overall = (list(values), [list(e) for e in edges])
    
    # Phase 3: Local search on values for best edge set
    if best_overall:
        values, edges = best_overall
        time_left = deadline - time.time()
        if time_left > 5:
            values, score = optimize_values_local(values, edges, rng, time.time() + min(time_left - 2, 20))
            if score < best_overall_score:
                best_overall_score = score
                best_overall = (list(values), [list(e) for e in edges])
    
    # Phase 4: More iterations if time permits
    if best_overall and time.time() < deadline - 15:
        for trial in range(20):
            if time.time() >= deadline - 10:
                break
            
            values = generate_structured_values(rng)
            edges = greedy_edges_for_values(values, rng)
            score = evaluate(values, edges)
            
            if score < best_overall_score:
                # Try to improve with local search
                time_left = deadline - time.time()
                if time_left > 10:
                    edges, score = local_search_edges(values, edges, rng, time.time() + min(time_left - 5, 15))
                
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall = (list(values), [list(e) for e in edges])
    
    # Ensure we have a valid result
    if best_overall is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_overall
    
    # Final validation
    assert len(values) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert all(1 <= v <= max_val for v in values)
    assert len(set(values)) == n
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        u, v = e
        assert isinstance(u, int) and isinstance(v, int)
        assert u != v
        assert 0 <= u < n and 0 <= v < n
        key = tuple(sorted((u, v)))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
