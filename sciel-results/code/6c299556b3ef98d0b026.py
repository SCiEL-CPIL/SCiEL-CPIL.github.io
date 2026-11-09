import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _evaluate_edges(values, edge_set):
    """
    Evaluate a specific set of edges.
    Returns (score, sums_set, prods_set)
    """
    sums = set()
    prods = set()
    for u, v in edge_set:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods)), sums, prods

def _optimize_edges_for_values(values, time_limit):
    """
    Find a set of 35 edges that minimizes the raw score for given values.
    Uses Simulated Annealing / Hill Climbing.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_data = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data[(u, v)] = (s, p)
        
    # Initialize with a random set of 35 edges
    current_edges = set(random.sample(all_edges, 35))
    
    # Calculate initial state
    current_score, current_sums, current_prods = _evaluate_edges(values, current_edges)
    # We need counts to update efficiently
    sum_counts = {}
    prod_counts = {}
    for u, v in current_edges:
        s, p = edge_data[(u, v)]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    best_score = current_score
    best_edges = set(current_edges)
    best_sum_counts = dict(sum_counts)
    best_prod_counts = dict(prod_counts)
    
    start_time = time.time()
    temp = 5.0
    
    while time.time() - start_time < time_limit:
        if not current_edges:
            break
            
        # Pick a random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick a random edge to add (not in current)
        # To save time, we can pick from a small random subset of non-edges
        non_edges = [e for e in all_edges if e not in current_edges]
        if not non_edges:
            break
        add_edge = random.choice(non_edges)
        
        # Perform swap
        current_edges.remove(remove_edge)
        current_edges.add(add_edge)
        
        s_rem, p_rem = edge_data[remove_edge]
        s_add, p_add = edge_data[add_edge]
        
        # Update counts
        sum_counts[s_rem] -= 1
        if sum_counts[s_rem] == 0:
            del sum_counts[s_rem]
        sum_counts[s_add] = sum_counts.get(s_add, 0) + 1
        
        prod_counts[p_rem] -= 1
        if prod_counts[p_rem] == 0:
            del prod_counts[p_rem]
        prod_counts[p_add] = prod_counts.get(p_add, 0) + 1
        
        new_score = max(len(sum_counts), len(prod_counts))
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
                best_sum_counts = dict(sum_counts)
                best_prod_counts = dict(prod_counts)
        else:
            # Revert
            current_edges.remove(add_edge)
            current_edges.add(remove_edge)
            
            sum_counts[s_add] -= 1
            if sum_counts[s_add] == 0:
                del sum_counts[s_add]
            sum_counts[s_rem] = sum_counts.get(s_rem, 0) + 1
            
            prod_counts[p_add] -= 1
            if prod_counts[p_add] == 0:
                del prod_counts[p_add]
            prod_counts[p_rem] = prod_counts.get(p_rem, 0) + 1
            
        temp *= 0.999
        if temp < 0.01:
            break
            
    return [list(e) for e in best_edges], best_score

def _generate_value_candidates():
    """
    Generate diverse sets of 14 distinct positive integers <= 10000.
    """
    candidates = []
    
    # 1. Small integers 1-14
    candidates.append(list(range(1, 15)))
    
    # 2. Multiples of 2
    candidates.append([i * 2 for i in range(1, 15)])
    
    # 3. Multiples of 3
    candidates.append([i * 3 for i in range(1, 15)])
    
    # 4. Geometric progression (powers of 2)
    candidates.append([2**i for i in range(1, 15)])
    
    # 5. Highly composite small numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36])
    
    # 6. Random small numbers
    random.seed(123)
    base = list(range(1, 50))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 7. Another random set
    random.seed(456)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 8. Arithmetic progression with diff 4
    candidates.append([1 + i*4 for i in range(14)])
    
    # 9. Mixed small primes and composites
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 10. Values centered around 10
    candidates.append([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    
    # 11. Values with many divisors
    candidates.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # 12. Random 1-20
    random.seed(789)
    base = list(range(1, 21))
    random.shuffle(base)
    candidates.append(base[:14])
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_candidates()
    
    # Shuffle value sets to avoid bias
    random.seed(100)
    random.shuffle(value_sets)
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Allocate time for edge optimization
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        # Give each value set a chunk of time
        # First few get more time
        edge_time = min(5.0, time_left / 2.0)
        
        edges, score = _optimize_edges_for_values(values, edge_time)
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {
                'values': values,
                'edges': edges
            }
            
        # Early exit if we hit a very low score
        if best_global_score <= 10:
            break

    if best_global_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {
            'values': values,
            'edges': edges
        }
        
    return best_global_result
