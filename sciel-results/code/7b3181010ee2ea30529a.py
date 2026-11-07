import time
import random
from itertools import combinations

def _score_fast(values, edges):
    """Fast score computation."""
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    n = 14
    m = 35
    all_possible = list(combinations(range(n), 2))
    
    # Helper to generate a random edge set of size m
    def random_edge_set():
        return random.sample(all_possible, m)
    
    # Helper to do a quick local search on edges
    def optimize_edges(values, initial_edges, max_iters=20):
        current_edges = list(initial_edges)
        current_score = _score_fast(values, current_edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try random swaps
            for _ in range(10):
                if time.time() > deadline:
                    break
                idx = random.randint(0, len(current_edges) - 1)
                # Pick a random edge not in current set
                other_edge = random.choice(all_possible)
                if other_edge in current_edges:
                    continue
                
                new_edges = current_edges.copy()
                new_edges[idx] = other_edge
                
                new_score = _score_fast(values, new_edges)
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    break
            
            if not improved:
                break
        
        return current_score, current_edges

    # Strategy 1: Value sets with common factors
    # We want values that produce few distinct sums and products.
    # Multiples of a small number help with products.
    # Arithmetic progressions help with sums.
    
    random.seed(42)
    
    candidate_value_sets = []
    
    # 1. Multiples of 2
    for trial in range(5):
        vals = []
        while len(vals) < n:
            v = random.randint(1, 5000) * 2
            if v not in vals and v <= 10000:
                vals.append(v)
        if len(set(vals)) == n:
            candidate_value_sets.append(vals)

    # 2. Multiples of 3
    for trial in range(5):
        vals = []
        while len(vals) < n:
            v = random.randint(1, 3333) * 3
            if v not in vals and v <= 10000:
                vals.append(v)
        if len(set(vals)) == n:
            candidate_value_sets.append(vals)

    # 3. Small integers (1-100) - sums are small, products are small
    for trial in range(5):
        vals = random.sample(range(1, 101), n)
        candidate_value_sets.append(vals)
        
    # 4. Arithmetic Progressions
    for base in range(1, 100, 10):
        for step in range(1, 20, 3):
            vals = [base + i * step for i in range(n)]
            if max(vals) <= 10000 and min(vals) >= 1 and len(set(vals)) == n:
                candidate_value_sets.append(vals)
                
    # Evaluate each candidate
    for vals in candidate_value_sets:
        if time.time() > deadline:
            break
            
        # Start with a few random edge sets
        for trial in range(3):
            if time.time() > deadline:
                break
            
            initial_edges = random_edge_set()
            score, edges = optimize_edges(vals, initial_edges, max_iters=10)
            
            if score < best_score:
                best_score = score
                best_values = vals
                best_edges = edges
    
    # Fallback if nothing found
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = all_possible[:m]
        best_score = _score_fast(best_values, best_edges)
        
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}

def construct():
    return _construct()
