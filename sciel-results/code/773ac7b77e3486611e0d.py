"""Construct a 14-node, 35-edge graph with 14 distinct positive ints <=10000 minimizing
max(#distinct edge sums, #distinct edge products)."""

import random
import math
from itertools import combinations


def _score(values, edges):
    if len(set(values)) != len(values):
        return 10**9
    sset = set()
    pset = set()
    for u, v in edges:
        sset.add(values[u] + values[v])
        pset.add(values[u] * values[v])
    return max(len(sset), len(pset))


def _fast_edge_optimization(values, num_edges=35, iters=5000, seed=42):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [values[u] + values[v] for u, v in all_edges]
    edge_prods = [values[u] * values[v] for u, v in all_edges]
    
    rng = random.Random(seed)
    
    # Initialize a random set of edges
    current_edges = set(rng.sample(range(num_all), num_edges))
    
    # Maintain counts of sums and products in current set
    sum_counts = {}
    prod_counts = {}
    for idx in current_edges:
        s = edge_sums[idx]
        p = edge_prods[idx]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    current_score = max(len(sum_counts), len(prod_counts))
    best_score = current_score
    best_edges = set(current_edges)
    
    for _ in range(iters):
        remove_idx = rng.choice(list(current_edges))
        add_idx = rng.randint(0, num_all - 1)
        if add_idx in current_edges:
            continue
            
        rs = edge_sums[remove_idx]
        rp = edge_prods[remove_idx]
        as_ = edge_sums[add_idx]
        ap = edge_prods[add_idx]
        
        # Try swap
        sum_counts[rs] -= 1
        if sum_counts[rs] == 0:
            del sum_counts[rs]
        prod_counts[rp] -= 1
        if prod_counts[rp] == 0:
            del prod_counts[rp]
            
        sum_counts[as_] = sum_counts.get(as_, 0) + 1
        prod_counts[ap] = prod_counts.get(ap, 0) + 1
        
        new_score = max(len(sum_counts), len(prod_counts))
        
        if new_score < current_score:
            current_edges.remove(remove_idx)
            current_edges.add(add_idx)
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                if best_score <= 14:
                    break
        else:
            # Revert
            sum_counts[as_] -= 1
            if sum_counts[as_] == 0:
                del sum_counts[as_]
            prod_counts[ap] -= 1
            if prod_counts[ap] == 0:
                del prod_counts[ap]
                
            sum_counts[rs] = sum_counts.get(rs, 0) + 1
            prod_counts[rp] = prod_counts.get(rp, 0) + 1
            
    best_edge_pairs = [list(all_edges[i]) for i in best_edges]
    return best_score, best_edge_pairs


def construct():
    # Simulated Annealing on values
    rng = random.Random(123)
    
    # Initial values: Small integers
    values = list(range(1, 15))
    
    # Get initial cost
    curr_score, curr_edges = _fast_edge_optimization(values, iters=2000, seed=42)
    best_score = curr_score
    best_values = values[:]
    best_edges = curr_edges
    
    T = 50.0
    min_T = 0.1
    
    # Run SA for a fixed number of steps
    # We need to be fast. _fast_edge_optimization takes ~0.1s?
    # 5000 iters * dict ops.
    # Let's estimate: 5000 swaps is fast.
    # We can do ~100-200 SA iterations in 60s.
    
    for i in range(150):
        # Perturb values
        new_values = values[:]
        # Change 1-2 values
        num_changes = rng.randint(1, 2)
        for _ in range(num_changes):
            idx = rng.randint(0, 13)
            # Random walk
            delta = rng.randint(-10, 10)
            new_val = new_values[idx] + delta
            # Clamp
            if new_val < 1:
                new_val = 1
            elif new_val > 10000:
                new_val = 10000
            new_values[idx] = new_val
            
        # Ensure distinct
        if len(set(new_values)) != 14:
            continue
            
        # Evaluate
        new_score, new_edges = _fast_edge_optimization(new_values, iters=2000, seed=42+i)
        
        # Acceptance
        if new_score < curr_score:
            curr_score = new_score
            values = new_values
            curr_edges = new_edges
            if curr_score < best_score:
                best_score = curr_score
                best_values = values[:]
                best_edges = curr_edges
        else:
            # Simulated annealing acceptance
            if T > min_T:
                delta = new_score - curr_score
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    curr_score = new_score
                    values = new_values
                    curr_edges = new_edges
        T *= 0.95
        
    # Final type conversion
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': values, 'edges': edges}
