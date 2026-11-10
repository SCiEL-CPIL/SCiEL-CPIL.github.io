"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    n = 14
    m = 35
    
    # Start with consecutive integers
    best_values = list(range(1, n + 1))
    # Start with first m edges from complete graph
    all_edges = list(combinations(range(n), 2))
    best_edges = [list(e) for e in all_edges[:m]]
    
    best_score = _score(best_values, best_edges)
    
    # Try different value sets
    random.seed(42)
    
    # Strategy 1: Try various arithmetic progressions and value sets
    candidate_value_sets = []
    
    # Consecutive integers
    candidate_value_sets.append(list(range(1, n + 1)))
    # Wider spacing
    for start in range(1, 50):
        for step in range(1, 20):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidate_value_sets.append(vals)
    
    # Random sets
    for _ in range(200):
        vals = sorted(random.sample(range(1, 10001), n))
        candidate_value_sets.append(vals)
    
    # Small clustered values
    for base in range(1, 100):
        vals = sorted(random.sample(range(base, base + 50), n))
        if len(set(vals)) == n and max(vals) <= 10000:
            candidate_value_sets.append(vals)
    
    # Evaluate initial candidates
    for vals in candidate_value_sets[:50]:
        if time.time() > deadline:
            break
        s = _score(vals, best_edges)
        if s < best_score:
            best_score = s
            best_values = list(vals)
    
    # Local search on values
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Randomly perturb one value
        new_values = list(best_values)
        idx = random.randint(0, n - 1)
        new_val = random.randint(1, 10000)
        if new_val in new_values:
            continue
        new_values[idx] = new_val
        if len(set(new_values)) != n:
            continue
        
        s = _score(new_values, best_edges)
        if s < best_score:
            best_score = s
            best_values = new_values
    
    # Local search on edges: try swapping edges
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random edge to remove and a random non-edge to add
        edge_to_remove = random.randint(0, m - 1)
        removed = tuple(sorted(best_edges[edge_to_remove]))
        
        # Find a non-edge
        edge_set = set(tuple(sorted(e)) for e in best_edges)
        attempts = 0
        while attempts < 50:
            u, v = random.sample(range(n), 2)
            new_edge = tuple(sorted((u, v)))
            if new_edge not in edge_set:
                break
            attempts += 1
        else:
            continue
        
        # Create new edge list
        new_edges = [list(e) for e in best_edges if tuple(sorted(e)) != removed]
        new_edges.append(list(new_edge))
        
        s = _score(best_values, new_edges)
        if s < best_score:
            best_score = s
            best_edges = new_edges
    
    # Final local search: try replacing individual values
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, n - 1)
        old_val = best_values[idx]
        
        # Try nearby values
        for delta in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50]:
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in best_values:
                continue
            
            new_values = list(best_values)
            new_values[idx] = new_val
            
            s = _score(new_values, best_edges)
            if s < best_score:
                best_score = s
                best_values = new_values
                break
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
