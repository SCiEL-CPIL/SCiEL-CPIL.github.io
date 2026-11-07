"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _find_best_edges(values):
    """
    Find a set of 35 edges that minimizes the score for given values.
    Uses a greedy heuristic with randomization.
    """
    n = len(values)
    m = P['m']
    all_pairs = list(combinations(range(n), 2))
    
    # Precompute sums and products for all pairs
    pair_props = {}
    for u, v in all_pairs:
        a, b = values[u], values[v]
        pair_props[(u, v)] = (a + b, a * b)
        
    best_edges = None
    best_score = float('inf')
    
    # Run greedy selection multiple times with random tie-breaking
    for _ in range(20):
        selected = []
        selected_set = set()
        current_sums = set()
        current_prods = set()
        
        for _ in range(m):
            best_edge = None
            best_metric = float('inf')
            best_sum_count = 0
            best_prod_count = 0
            best_new_sum = None
            best_new_prod = None
            
            candidates = []
            for e in all_pairs:
                if e in selected_set:
                    continue
                s, p = pair_props[e]
                
                # Calculate new counts if we add this edge
                new_sums = current_sums | {s}
                new_prods = current_prods | {p}
                
                # Metric: max of new counts
                # We prefer edges that don't increase the count, or increase it less
                # Also, we might prefer edges that add to existing sets if possible
                # Let's just use the resulting max count as primary metric
                metric = max(len(new_sums), len(new_prods))
                
                # Tie-breaking: prefer edges that add fewer new distinct values
                new_s = 1 if s not in current_sums else 0
                new_p = 1 if p not in current_prods else 0
                tie_breaker = new_s + new_p
                
                if metric < best_metric or (metric == best_metric and tie_breaker < best_sum_count + best_prod_count):
                    best_metric = metric
                    best_edge = e
                    best_sum_count = new_s
                    best_prod_count = new_p
                    best_new_sum = s
                    best_new_prod = p
            if best_edge is None:
                break
            
            # Randomization: if there are other edges with same metric, pick randomly
            # For simplicity, we just pick the best found. To add diversity, we can randomize the order of candidates
            # But 20 restarts should cover it.
            
            selected.append(list(best_edge))
            selected_set.add(best_edge)
            current_sums.add(best_new_sum)
            current_prods.add(best_new_prod)
            
        if len(selected) == m:
            score = max(len(current_sums), len(current_prods))
            if score < best_score:
                best_score = score
                best_edges = selected
                
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    # Initial values: 1..14
    current_values = list(range(1, n + 1))
    current_edges, current_score = _find_best_edges(current_values)
    
    best_values = current_values.copy()
    best_edges = current_edges.copy()
    best_score = current_score
    
    # Simulated Annealing on values
    temperature = 100.0
    cooling_rate = 0.95
    min_temperature = 1.0
    
    while time.time() < deadline and temperature > min_temperature:
        # Perturb values
        new_values = current_values.copy()
        
        # Randomly change 1-3 values
        num_changes = random.randint(1, 3)
        for _ in range(num_changes):
            idx = random.randint(0, n - 1)
            # Pick a new random value
            new_val = random.randint(1, max_val)
            if new_val not in new_values:
                new_values[idx] = new_val
        
        # Check distinctness
        if len(set(new_values)) != n:
            continue
            
        # Find best edges for these new values
        new_edges, new_score = _find_best_edges(new_values)
        
        if new_score is None:
            continue
            
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values.copy()
                best_edges = current_edges.copy()
                
        temperature *= cooling_rate
        
    # Final polish: Local search on values with fixed edges? Or re-optimize edges for best values?
    # Re-optimize edges for the best values found
    final_edges, final_score = _find_best_edges(best_values)
    if final_score < best_score:
        best_score = final_score
        best_edges = final_edges
        
    # Validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_tuples = set()
    for u, v in best_edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n and 0 <= v < n
        assert u != v
        t = (min(u, v), max(u, v))
        assert t not in edge_tuples
        edge_tuples.add(t)
        
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
