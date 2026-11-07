"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _evaluate(values):
    """
    Given a list of 14 distinct integers, find the 35 edges that minimize
    max(num_distinct_sums, num_distinct_products).
    Returns the minimum max value found by greedy selection.
    """
    n = len(values)
    m = 35
    
    # Pre-calculate sums and products for all pairs
    pairs = list(itertools.combinations(range(n), 2))
    edge_data = []
    for i, j in pairs:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
    
    # Greedy selection
    selected_edges = []
    current_sums = set()
    current_prods = set()
    
    # We want to pick edges that minimize the growth of distinct sums and products.
    # Heuristic: pick edge that adds 0 new sums and 0 new prods if possible.
    # Then 0 new sums, 1 new prod, etc.
    # Score: max(len(current_sums) + add_s, len(current_prods) + add_p)
    
    available = list(range(len(edge_data)))
    
    for _ in range(m):
        best_idx = -1
        best_score = float('inf')
        best_add_s = 0
        best_add_p = 0
        
        for idx in available:
            i, j, s, p = edge_data[idx]
            add_s = 0 if s in current_sums else 1
            add_p = 0 if p in current_prods else 1
            
            new_s_count = len(current_sums) + add_s
            new_p_count = len(current_prods) + add_p
            
            # We want to minimize the max of the final counts.
            # A good proxy is the current max after adding this edge.
            score = max(new_s_count, new_p_count)
            
            # Tie-breaking: prefer edges that add fewer total new items
            # and then prefer edges with smaller sums/products? 
            # Actually, just minimizing the max is the primary goal.
            # Secondary: minimize add_s + add_p to keep sets small for future steps.
            
            if score < best_score:
                best_score = score
                best_idx = idx
                best_add_s = add_s
                best_add_p = add_p
            elif score == best_score:
                # Tie-breaker: prefer edge that adds fewer new distinct values in total
                if (add_s + add_p) < (best_add_s + best_add_p):
                    best_idx = idx
                    best_add_s = add_s
                    best_add_p = add_p
        
        if best_idx == -1:
            break
            
        i, j, s, p = edge_data[best_idx]
        selected_edges.append([i, j])
        current_sums.add(s)
        current_prods.add(p)
        available.remove(best_idx)
        
    return max(len(current_sums), len(current_prods))

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin
    
    # Initial values: small integers are good for collisions
    best_values = list(range(1, n + 1))
    best_score = _evaluate(best_values)
    
    # Local Search
    # We will perturb one value at a time.
    # Range of values: 1 to 10000. But small values are better for collisions.
    # Let's restrict search space to 1..50 initially, then expand if needed?
    # Actually, 1..14 is already quite good. Let's try random small integers.
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # Pick a random index to change
        idx = random.randint(0, n - 1)
        # Pick a new value
        # Bias towards small numbers
        new_val = random.randint(1, 30)
        
        # Ensure distinctness
        if new_val in best_values and best_values[idx] != new_val:
            continue
            
        old_val = best_values[idx]
        best_values[idx] = new_val
        
        current_score = _evaluate(best_values)
        
        if current_score < best_score:
            best_score = current_score
            # Optionally, we can try to refine further
        else:
            # Revert
            best_values[idx] = old_val
    
    # Final evaluation to get the edges for the best values
    # Re-run the greedy logic to get the actual edges
    values = best_values
    pairs = list(itertools.combinations(range(n), 2))
    edge_data = []
    for i, j in pairs:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
        
    selected_edges = []
    current_sums = set()
    current_prods = set()
    available = list(range(len(edge_data)))
    
    for _ in range(m):
        best_idx = -1
        best_score = float('inf')
        best_add_s = 0
        best_add_p = 0
        
        for idx in available:
            i, j, s, p = edge_data[idx]
            add_s = 0 if s in current_sums else 1
            add_p = 0 if p in current_prods else 1
            
            new_s_count = len(current_sums) + add_s
            new_p_count = len(current_prods) + add_p
            
            score = max(new_s_count, new_p_count)
            
            if score < best_score:
                best_score = score
                best_idx = idx
                best_add_s = add_s
                best_add_p = add_p
            elif score == best_score:
                if (add_s + add_p) < (best_add_s + best_add_p):
                    best_idx = idx
                    best_add_s = add_s
                    best_add_p = add_p
        
        if best_idx == -1:
            break
            
        i, j, s, p = edge_data[best_idx]
        selected_edges.append([i, j])
        current_sums.add(s)
        current_prods.add(p)
        available.remove(best_idx)
        
    return {'values': values, 'edges': selected_edges}
# EVOLVE-BLOCK-END
