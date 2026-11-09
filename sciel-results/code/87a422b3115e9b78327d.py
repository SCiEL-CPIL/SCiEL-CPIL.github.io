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
    Fast evaluation using pre-computed lookup or direct calculation.
    Since n=14 is small, direct calculation with sets is fast enough, 
    but we can optimize by tracking counts if we were doing many iterations.
    For SA, we just need the score.
    """
    sums = set()
    prods = set()
    for u, v in edges_set:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _generate_value_candidates():
    """
    Generate diverse sets of 14 distinct positive integers <= 10000.
    Focus on numbers that create additive and multiplicative collisions.
    """
    candidates = []
    
    # 1. Small integers 1-14 (high additive collision potential)
    candidates.append(list(range(1, 15)))
    
    # 2. Multiples of a common base (e.g., 1..14 * 2)
    candidates.append([i * 2 for i in range(1, 15)])
    
    # 3. Mix of small primes and composites
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 4. Numbers with high factor overlap (multiples of 6)
    candidates.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # 5. Random small numbers
    random.seed(123)
    base = list(range(1, 50))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 6. Another random set
    random.seed(456)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    return candidates

def _simulated_annealing(values, n, m, initial_edges, time_limit):
    """
    Optimize edge selection using Simulated Annealing.
    Moves: Swap an existing edge with a non-existing edge.
    """
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    # Start from initial edges
    current_edges = set(tuple(sorted(e)) for e in initial_edges)
    current_score = _evaluate_state(values, current_edges)
    
    best_edges = set(current_edges)
    best_score = current_score
    
    # Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    cooling_rate = 0.999
    
    temp = start_temp
    start_time = time.time()
    
    while temp > end_temp:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random edge to remove
        edges_list = list(current_edges)
        if not edges_list:
            break
        remove_edge = random.choice(edges_list)
        
        # Pick a random edge to add (must not be in current set)
        # To avoid picking the same edge, we can sample from all_edges
        # or maintain a list of non-edges. With 91 total edges and 35 selected,
        # 56 are non-edges. Sampling is fine.
        add_edge = random.choice(all_edges)
        if add_edge in current_edges:
            # Retry a few times or just skip
            continue
            
        # Perform swap
        current_edges.remove(remove_edge)
        current_edges.add(add_edge)
        
        new_score = _evaluate_state(values, current_edges)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Reject: swap back
            current_edges.remove(add_edge)
            current_edges.add(remove_edge)
            
        temp *= cooling_rate
        
    return [list(e) for e in best_edges], best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit)
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_candidates()
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Initial guess: Random 35 edges
        all_edges = list(combinations(range(14), 2))
        
        # Run SA with multiple restarts if time permits
        # We allocate time per value set
        time_per_set = (deadline - time.time()) / max(1, len(value_sets))
        
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initial edges
            initial_edges = random.sample(all_edges, 35)
            
            # Run SA for a portion of the time allocated
            sa_time = time_per_set / 5.0
            if sa_time < 0.5:
                sa_time = 0.5
                
            edges, score = _simulated_annealing(values, 14, 35, initial_edges, sa_time)
            
            # Validate
            if len(edges) != 35:
                continue
            edge_tuples = set(tuple(sorted(e)) for e in edges)
            if len(edge_tuples) != 35:
                continue
            if any(u == v for u, v in edges):
                continue
                
            if score < best_global_score:
                best_global_score = score
                best_global_result = {
                    'values': values,
                    'edges': edges
                }
                
                # Early exit if perfect (theoretical min is likely higher than 1)
                if best_global_score <= 10:
                    break
                    
        if time.time() > deadline:
            break
            
    if best_global_result is None:
        # Fallback
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    return best_global_result
# EVOLVE-BLOCK-END
