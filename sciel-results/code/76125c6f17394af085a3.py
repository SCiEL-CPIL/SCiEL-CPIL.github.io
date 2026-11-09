"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _evaluate_state(values, edges_set):
    """
    Fast evaluation using sets.
    """
    sums = set()
    prods = set()
    for u, v in edges_set:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _generate_value_structures():
    """
    Generate diverse sets of 14 distinct positive integers <= 10000.
    Focus on structures that maximize additive and multiplicative collisions.
    """
    structures = []
    
    # 1. Powers of 2 (excellent multiplicative structure, bad additive but bounded)
    # 1, 2, 4, ..., 8192 (14 values)
    structures.append([2**i for i in range(14)])
    
    # 2. Multiples of a small highly composite number (e.g., 60)
    # 60, 120, ..., 840
    structures.append([60 * i for i in range(1, 15)])
    
    # 3. Small integers 1-14 (maximum additive collision potential)
    structures.append(list(range(1, 15)))
    
    # 4. Multiples of 10 (10, 20, ..., 140)
    structures.append([10 * i for i in range(1, 15)])
    
    # 5. "Almost" powers of 2 with slight perturbation to break some additive symmetries
    # while keeping multiplicative structure loose
    base_powers = [2**i for i in range(14)]
    structures.append([max(1, x + random.randint(-1, 1)) for x in base_powers])
    
    # 6. Random small numbers with high density (1-30)
    random.seed(42)
    candidates = list(range(1, 31))
    random.shuffle(candidates)
    structures.append(sorted(candidates[:14]))
    
    # 7. Multiples of 4 (4, 8, ..., 56)
    structures.append([4 * i for i in range(1, 15)])
    
    # 8. A mix of small primes and composites optimized for small products
    structures.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    return structures

def _greedy_edge_selection_with_tracking(values, n, m, time_limit):
    """
    Greedy edge selection that tracks marginal cost.
    Strategy: Start with empty graph. At each step, add the edge that causes
    the smallest increase in max(distinct_sums, distinct_prods).
    To handle ties and avoid local minima, we use a randomized tie-breaking
    and occasionally look 1-step ahead (2-opt style) if time permits.
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    current_sums = set()
    current_prods = set()
    selected_edges = []
    
    start_time = time.time()
    
    for step in range(m):
        if time.time() - start_time > time_limit:
            break
            
        best_edge = None
        best_score = float('inf')
        best_sums = None
        best_prods = None
        best_marginal = float('inf')
        
        # We evaluate all remaining edges to find the one with minimal marginal cost.
        # With n=14, there are 91 edges. At each step we check up to 91 edges.
        # Cost per edge: O(1) set check + O(1) add.
        # Total cost: 35 * 91 * O(1) which is very fast.
        
        # Optimization: Pre-compute sum and product for all pairs?
        # Since values are fixed for this call, we can pre-compute.
        # But we are inside the loop for a specific value set.
        
        # To speed up, we can just iterate.
        for u, v in all_edges:
            if (u, v) in selected_edges or (v, u) in selected_edges:
                continue
                
            su = values[u]
            sv = values[v]
            s_val = su + sv
            p_val = su * sv
            
            # Calculate marginal cost
            # If s_val is already in sums, marginal sum cost is 0, else 1
            # If p_val is already in prods, marginal prod cost is 0, else 1
            
            # We want to minimize the resulting max(len(sums), len(prods))
            # Let current_max = max(len(current_sums), len(current_prods))
            
            new_sum_count = len(current_sums) + (1 if s_val not in current_sums else 0)
            new_prod_count = len(current_prods) + (1 if p_val not in current_prods else 0)
            
            new_score = max(new_sum_count, new_prod_count)
            
            # We prioritize edges that don't increase the score.
            # If multiple edges don't increase the score, we prefer those that
            # add to the "smaller" set to balance them, or just randomize.
            
            if new_score < best_score:
                best_score = new_score
                best_edge = (u, v)
                best_sums = s_val
                best_prods = p_val
            elif new_score == best_score:
                # Tie-breaking: Prefer edges that add to the currently smaller set?
                # Or just randomize to escape symmetry.
                if random.random() < 0.5:
                    best_edge = (u, v)
                    best_sums = s_val
                    best_prods = p_val
        
        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        current_sums.add(best_sums)
        current_prods.add(best_prods)
        
        # If we hit a score that is theoretically very low, we might want to 
        # do a local search later, but for now, greedy is fast.
        
    return [list(e) for e in selected_edges]

def _local_refinement(values, edges, time_limit):
    """
    Local search to improve the greedy result.
    Moves: Swap an existing edge with a non-existing edge.
    """
    all_edges = list(combinations(range(14), 2))
    all_edges_set = set(all_edges)
    
    current_edges = set(tuple(sorted(e)) for e in edges)
    current_score = _evaluate_state(values, current_edges)
    
    best_edges = set(current_edges)
    best_score = current_score
    
    start_time = time.time()
    temp = 1.0
    
    while time.time() - start_time < time_limit:
        if temp < 0.01:
            break
            
        # Pick a random edge to remove
        edges_list = list(current_edges)
        if not edges_list:
            break
        remove_edge = random.choice(edges_list)
        
        # Pick a random edge to add
        add_edge = random.choice(all_edges)
        if add_edge in current_edges:
            continue
            
        # Perform swap
        current_edges.remove(remove_edge)
        current_edges.add(add_edge)
        
        new_score = _evaluate_state(values, current_edges)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Reject
            current_edges.remove(add_edge)
            current_edges.add(remove_edge)
            
        temp *= 0.999
        
    return [list(e) for e in best_edges], best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_structures = _generate_value_structures()
    
    # Shuffle structures to ensure diversity in time allocation
    random.shuffle(value_structures)
    
    for values in value_structures:
        if time.time() > deadline:
            break
            
        # Ensure types are correct and constraints met
        values = [int(v) for v in values]
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # 1. Greedy Selection (Fast, good initial structure)
        greedy_time = 1.0
        edges = _greedy_edge_selection_with_tracking(values, 14, 35, greedy_time)
        
        if len(edges) != 35:
            # Fallback to random if greedy failed (shouldn't happen)
            all_edges = list(combinations(range(14), 2))
            edges = random.sample(all_edges, 35)
            
        score = _evaluate_state(values, set(tuple(sorted(e)) for e in edges))
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {
                'values': values,
                'edges': edges
            }
            
        # 2. Local Refinement (If time permits and score is promising)
        # We allocate remaining time to local search for this value set
        remaining_time = deadline - time.time()
        # Allocate a chunk to local search, e.g., 2 seconds per value set
        local_time = min(2.0, remaining_time / 2.0)
        
        if local_time > 0.5:
            refined_edges, refined_score = _local_refinement(values, edges, local_time)
            
            if refined_score < best_global_score:
                best_global_score = refined_score
                best_global_result = {
                    'values': values,
                    'edges': refined_edges
                }
                
            # If we found a very good score, we can stop early
            if best_global_score <= 12:
                break

    if best_global_result is None:
        # Fallback
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    # Final type checking and conversion
    res_values = [int(v) for v in best_global_result['values']]
    res_edges = [[int(u), int(v)] for u, v in best_global_result['edges']]
    
    return {
        'values': res_values,
        'edges': res_edges
    }
# EVOLVE-BLOCK-END
