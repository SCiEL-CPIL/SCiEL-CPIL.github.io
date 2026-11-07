"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    # 1. Search for optimal values
    # Objective: Minimize max(distinct_sums_all, distinct_products_all)
    
    def evaluate_values(vals):
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i + 1, n):
                sums.add(vals[i] + vals[j])
                prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))
        
    # Generate initial candidates
    candidates = []
    
    # 1. Arithmetic Progression
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(10, 10 + n)))
    
    # 2. Powers of 2 and 3 mix (Heuristic for low product distinctness)
    # We want exponents to be small and clustered
    exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (3,0), (0,2), (2,1), (4,0), (1,2), (3,1), (0,3), (5,0), (2,2)]
    vals_23 = [ (2**a) * (3**b) for a, b in exps ]
    if len(set(vals_23)) == n and max(vals_23) <= 10000:
        candidates.append(vals_23)
        
    # 3. Random candidates
    for _ in range(20):
        vals = random.sample(range(1, 10001), n)
        candidates.append(vals)
        
    best_vals = None
    best_score = float('inf')
    
    # Local Search
    iterations = 0
    while time.time() - start_time < time_limit * 0.7:
        # Pick a random candidate or the best
        if best_vals and random.random() < 0.5:
            current = list(best_vals)
        else:
            current = list(random.choice(candidates))
            
        # Ensure distinct and valid
        if len(set(current)) != n or min(current) < 1 or max(current) > 10000:
            continue
            
        score = evaluate_values(current)
        if score < best_score:
            best_score = score
            best_vals = list(current)
            
        # Perturb
        idx = random.randint(0, n - 1)
        old_val = current[idx]
        
        # Try different types of perturbations
        r = random.random()
        if r < 0.3:
            # Small step
            step = random.randint(-5, 5)
            new_val = old_val + step
        elif r < 0.6:
            # Medium step
            new_val = random.randint(max(1, old_val - 50), min(10000, old_val + 50))
        else:
            # Random jump
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000 or new_val in current and current.index(new_val) != idx:
            continue
            
        current[idx] = new_val
        
        if len(set(current)) != n:
            continue
            
        new_score = evaluate_values(current)
        
        # Accept if better or with small probability (Simulated Annealing-like)
        if new_score < score:
            # Keep the perturbed version in candidates pool? 
            # Just update best if it's the best
            if new_score < best_score:
                best_score = new_score
                best_vals = list(current)
        else:
            # Revert for next iteration if not accepted? 
            # We are using a "best so far" approach, so we don't need to revert 
            # if we just evaluate the new state. 
            # But we need to manage the 'current' state for the next perturbation.
            # Let's stick to a simple hill climbing on the best candidate.
            pass

    if best_vals is None:
        best_vals = list(range(1, n + 1))

    # 2. Select edges
    # Greedy selection to minimize actual distinct sums/products for the 35 edges
    values = best_vals
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    current_sums = set()
    current_products = set()
    selected_edges = []
    remaining_edges = set(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_score_edge = float('inf')
        best_new_sums = set()
        best_new_products = set()
        
        for edge in remaining_edges:
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            
            new_sums = current_sums | {s}
            new_products = current_products | {p}
            
            score = max(len(new_sums), len(new_products))
            
            if score < best_score_edge:
                best_score_edge = score
                best_edge = edge
                best_new_sums = new_sums
                best_new_products = new_products
            elif score == best_score_edge:
                # Tie-breaker: prefer edges that don't add new products
                # or add fewer new products
                new_p_count = len(new_products - current_products)
                old_p_count = len(best_new_products - current_products)
                if new_p_count < old_p_count:
                    best_edge = edge
                    best_new_sums = new_sums
                    best_new_products = new_products
                elif new_p_count == old_p_count:
                    new_s_count = len(new_sums - current_sums)
                    old_s_count = len(best_new_sums - current_sums)
                    if new_s_count < old_s_count:
                        best_edge = edge
                        best_new_sums = new_sums
                        best_new_products = new_products
                        
        if best_edge is None:
            break
            
        selected_edges.append(list(best_edge))
        current_sums = best_new_sums
        current_products = best_new_products
        remaining_edges.remove(best_edge)
        
    return {'values': values, 'edges': selected_edges}
# EVOLVE-BLOCK-END
