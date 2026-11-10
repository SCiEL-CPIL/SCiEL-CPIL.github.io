"""Construct a 14-node, 35-edge graph with 14 distinct positive ints <=10000 minimizing
max(#distinct edge sums, #distinct edge products)."""

import random
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


def _optimize_edges_for_values(values, num_edges=35, iters=50000, rng=random.Random(0)):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = []
    edge_prods = []
    for u, v in all_edges:
        edge_sums.append(values[u] + values[v])
        edge_prods.append(values[u] * values[v])
        
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
        
    current_distinct_sums = len(sum_counts)
    current_distinct_prods = len(prod_counts)
    current_score = max(current_distinct_sums, current_distinct_prods)
    
    best_score = current_score
    best_edges = set(current_edges)
    
    for _ in range(iters):
        # Pick a random edge to remove
        remove_idx = rng.choice(list(current_edges))
        
        # Pick a random edge to add (not in current set)
        # To speed up, we can pick from all edges and check if it's in set, 
        # or maintain a list of available edges. 
        # Given 91 edges, picking random and checking is fine.
        add_idx = rng.randint(0, num_all - 1)
        if add_idx in current_edges:
            continue
            
        # Calculate potential new state
        # Remove remove_idx
        rs = edge_sums[remove_idx]
        rp = edge_prods[remove_idx]
        
        sum_counts[rs] -= 1
        if sum_counts[rs] == 0:
            del sum_counts[rs]
        else:
            pass # just decrement
            
        prod_counts[rp] -= 1
        if prod_counts[rp] == 0:
            del prod_counts[rp]
            
        # Add add_idx
        as_ = edge_sums[add_idx]
        ap = edge_prods[add_idx]
        
        sum_counts[as_] = sum_counts.get(as_, 0) + 1
        prod_counts[ap] = prod_counts.get(ap, 0) + 1
        
        new_distinct_sums = len(sum_counts)
        new_distinct_prods = len(prod_counts)
        new_score = max(new_distinct_sums, new_distinct_prods)
        
        if new_score < current_score:
            # Accept move
            current_edges.remove(remove_idx)
            current_edges.add(add_idx)
            current_score = new_score
            current_distinct_sums = new_distinct_sums
            current_distinct_prods = new_distinct_prods
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
                if best_score <= 14: # Theoretical minimum is low, 14 is a good target
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
            
    # Convert best_edges indices back to (u, v) pairs
    best_edge_pairs = [list(all_edges[i]) for i in best_edges]
    return values, best_edge_pairs, best_score


def construct():
    # Define several candidate value sets
    # 1. Small integers
    vals1 = list(range(1, 15))
    
    # 2. Geometric progression (small base)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # Products will be huge, sums might collide? No, sums are distinct in GP.
    # So GP is bad for sums.
    
    # 3. Arithmetic progression
    vals3 = list(range(1, 29, 2)) # 1, 3, ..., 27
    
    # 4. Random small values
    rng_init = random.Random(123)
    vals4 = rng_init.sample(range(1, 50), 14)
    
    # 5. Values with many factors
    # 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84
    vals5 = [6*i for i in range(1, 15)]
    
    candidates = [vals1, vals3, vals4, vals5]
    
    best_values = None
    best_edges = None
    best_score = 10**9
    
    for vals in candidates:
        # Ensure distinctness and bounds
        if len(set(vals)) != 14 or any(v > 10000 or v <= 0 for v in vals):
            continue
            
        # Run optimization for this value set
        # Use multiple seeds for edge search
        for seed in [0, 1, 2, 3]:
            rng = random.Random(seed)
            opt_vals, opt_edges, score = _optimize_edges_for_values(vals, num_edges=35, iters=20000, rng=rng)
            
            if score < best_score:
                best_score = score
                best_values = opt_vals
                best_edges = opt_edges
                
                if best_score <= 14:
                    break
        if best_score <= 14:
            break
            
    if best_values is None:
        # Fallback
        best_values = vals1
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_score = _score(best_values, best_edges)
        
    # Final type conversion
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': values, 'edges': edges}
