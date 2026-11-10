"""Construct a 14-node, 35-edge graph with 14 distinct positive ints <=10000 minimizing
max(#distinct edge sums, #distinct edge products)."""

import random
import math
from itertools import combinations


def _greedy_edge_selection(values, num_edges=35, rng=random.Random(0)):
    """
    Selects num_edges edges from the complete graph on values to minimize
    max(distinct_sums, distinct_products) using a greedy heuristic.
    Used as a fast cost estimator for Simulated Annealing.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        edge_sums[(u, v)] = values[u] + values[v]
        edge_prods[(u, v)] = values[u] * values[v]
        
    selected_edges = []
    sum_counts = {}
    prod_counts = {}
    
    remaining_edges = set(all_edges)
    
    for _ in range(num_edges):
        best_edge = None
        best_score = float('inf')
        
        for e in remaining_edges:
            s = edge_sums[e]
            p = edge_prods[e]
            
            new_sum_count = 1 if s not in sum_counts else 0
            new_prod_count = 1 if p not in prod_counts else 0
            
            current_distinct_sums = len(sum_counts)
            current_distinct_prods = len(prod_counts)
            
            proj_sums = current_distinct_sums + new_sum_count
            proj_prods = current_distinct_prods + new_prod_count
            
            score = max(proj_sums, proj_prods)
            
            if score < best_score:
                best_score = score
                best_edge = e
            elif score == best_score and best_edge is not None:
                # Random tie break to escape local minima
                if rng.random() < 0.5:
                    best_edge = e
                    
        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        remaining_edges.remove(best_edge)
        
        s = edge_sums[best_edge]
        p = edge_prods[best_edge]
        
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    return selected_edges, max(len(sum_counts), len(prod_counts))


def _optimize_edges_for_values(values, num_edges=35, iters=50000, rng=random.Random(0)):
    """
    Local search on edges for a fixed set of values.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    edge_sums = []
    edge_prods = []
    for u, v in all_edges:
        edge_sums.append(values[u] + values[v])
        edge_prods.append(values[u] * values[v])
        
    # Initialize with greedy selection to get a good starting point
    init_edges, _ = _greedy_edge_selection(values, num_edges, rng)
    # Convert init_edges (tuples) to indices in all_edges
    edge_index_map = {e: i for i, e in enumerate(all_edges)}
    current_edges = set(edge_index_map[e] for e in init_edges)
    
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
        
        sum_counts[rs] -= 1
        if sum_counts[rs] == 0:
            del sum_counts[rs]
            
        prod_counts[rp] -= 1
        if prod_counts[rp] == 0:
            del prod_counts[rp]
            
        as_ = edge_sums[add_idx]
        ap = edge_prods[add_idx]
        
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
        else:
            sum_counts[as_] -= 1
            if sum_counts[as_] == 0:
                del sum_counts[as_]
                
            prod_counts[ap] -= 1
            if prod_counts[ap] == 0:
                del prod_counts[ap]
                
            sum_counts[rs] = sum_counts.get(rs, 0) + 1
            prod_counts[rp] = prod_counts.get(rp, 0) + 1
            
    best_edge_pairs = [list(all_edges[i]) for i in best_edges]
    return best_edge_pairs, best_score


def construct():
    n = 14
    max_val = 10000
    rng = random.Random(42)
    
    # Initialize with a random set of distinct values
    current_values = rng.sample(range(1, max_val + 1), n)
    
    # Fast evaluation using greedy
    _, current_score = _greedy_edge_selection(current_values, rng=rng)
    
    best_values = list(current_values)
    best_score = current_score
    
    # Simulated Annealing on values
    T = 1000.0
    cooling_rate = 0.995
    min_T = 1.0
    steps = 3000 # Adjusted for time
    
    for step in range(steps):
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        new_values = list(current_values)
        idx = rng.randint(0, n - 1)
        old_val = new_values[idx]
        
        step_size = rng.randint(1, 100)
        delta = rng.choice([-1, 1]) * step_size
        new_val = old_val + delta
        
        if new_val < 1:
            new_val = 1
        elif new_val > max_val:
            new_val = max_val
            
        if new_val in new_values[:idx] or new_val in new_values[idx+1:]:
            continue
            
        new_values[idx] = new_val
        
        _, new_score = _greedy_edge_selection(new_values, rng=rng)
        
        delta_score = new_score - current_score
        if delta_score < 0 or (T > 0 and rng.random() < math.exp(-delta_score / T)):
            current_values = new_values
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = list(current_values)
                
                if best_score <= 15:
                    break
        else:
            pass
            
    # Refine edges for the best values found
    final_edges, final_score = _optimize_edges_for_values(best_values, iters=50000, rng=rng)
    
    # Final type conversion
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in final_edges]
    
    return {'values': values, 'edges': edges}
