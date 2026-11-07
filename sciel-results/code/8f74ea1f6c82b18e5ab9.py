import time
import random
from itertools import combinations
from collections import Counter

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _mip_solve_edge_selection(values, m=35, deadline=None):
    """
    Exact MILP formulation to minimize max(distinct_sums, distinct_products).
    Uses pulp if available, falls back to heuristic if not.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [0] * num_all_edges
    edge_prods = [0] * num_all_edges
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
    
    # Get unique sum and product values
    unique_sums = sorted(set(edge_sums))
    unique_prods = sorted(set(edge_prods))
    
    # Map values to indices
    sum_to_idx = {v: i for i, v in enumerate(unique_sums)}
    prod_to_idx = {v: i for i, v in enumerate(unique_prods)}
    
    try:
        import pulp
        
        # Create MILP problem
        prob = pulp.LpProblem("edge_selection", pulp.LpMinimize)
        
        # Decision variables: x[e] = 1 if edge e is selected
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(num_all_edges)]
        
        # Variables for whether each sum/product value is used
        y_sum = [pulp.LpVariable(f"ys_{i}", cat='Binary') for i in range(len(unique_sums))]
        y_prod = [pulp.LpVariable(f"yp_{i}", cat='Binary') for i in range(len(unique_prods))]
        
        # Objective: minimize max(sum of y_sum, sum of y_prod)
        # We use a big-M trick: minimize T where T >= sum(y_sum) and T >= sum(y_prod)
        T = pulp.LpVariable("T", lowBound=0)
        
        prob += T
        prob += T >= pulp.lpSum(y_sum)
        prob += T >= pulp.lpSum(y_prod)
        
        # Constraint: select exactly m edges
        prob += pulp.lpSum(x) == m
        
        # Link x and y_sum: if edge e is selected, y_sum for its sum must be 1
        for e in range(num_all_edges):
            s_idx = sum_to_idx[edge_sums[e]]
            prob += x[e] <= y_sum[s_idx]
        
        # Link x and y_prod: if edge e is selected, y_prod for its product must be 1
        for e in range(num_all_edges):
            p_idx = prod_to_idx[edge_prods[e]]
            prob += x[e] <= y_prod[p_idx]
        
        # Solve with time limit
        remaining_time = 30.0 if deadline is None else max(1.0, deadline - time.time())
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(remaining_time)))
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            selected_edges = []
            for e in range(num_all_edges):
                if x[e].value() > 0.5:
                    selected_edges.append(all_edges[e])
            
            raw = _raw_value(values, selected_edges)
            return selected_edges, raw
        else:
            # Fallback to heuristic
            return _heuristic_edge_optimize(values, m, deadline)
            
    except ImportError:
        # No pulp available, use heuristic
        return _heuristic_edge_optimize(values, m, deadline)
    except Exception:
        # Fallback to heuristic
        return _heuristic_edge_optimize(values, m, deadline)

def _heuristic_edge_optimize(values, m=35, deadline=None):
    """
    Fallback heuristic edge optimization using local search.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [0] * num_all_edges
    edge_prods = [0] * num_all_edges
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
    
    best_score = float('inf')
    best_edges = None
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
            
        # Random initialization
        current_edges = set(random.sample(range(num_all_edges), m))
        
        # Initialize counters
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in current_edges:
            sum_counts[edge_sums[idx]] += 1
            prod_counts[edge_prods[idx]] += 1
            
        current_distinct_sums = len(sum_counts)
        current_distinct_prods = len(prod_counts)
        current_score = max(current_distinct_sums, current_distinct_prods)
        
        # Local Search
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations and (not deadline or time.time() < deadline):
            improved = False
            iteration += 1
            
            # Shuffle edges to avoid bias
            current_list = list(current_edges)
            outside_indices = [idx for idx in range(num_all_edges) if idx not in current_edges]
            random.shuffle(current_list)
            random.shuffle(outside_indices)
            
            for e_in_idx in current_list:
                s_in = edge_sums[e_in_idx]
                p_in = edge_prods[e_in_idx]
                
                for e_out_idx in outside_indices:
                    s_out = edge_sums[e_out_idx]
                    p_out = edge_prods[e_out_idx]
                    
                    # Simulate swap
                    sum_counts[s_in] -= 1
                    prod_counts[p_in] -= 1
                    
                    new_distinct_sums = current_distinct_sums
                    new_distinct_prods = current_distinct_prods
                    
                    if sum_counts[s_in] == 0:
                        new_distinct_sums -= 1
                    if prod_counts[p_in] == 0:
                        new_distinct_prods -= 1
                        
                    sum_counts[s_out] += 1
                    prod_counts[p_out] += 1
                    
                    if sum_counts[s_out] == 1:
                        new_distinct_sums += 1
                    if prod_counts[p_out] == 1:
                        new_distinct_prods += 1
                        
                    new_score = max(new_distinct_sums, new_distinct_prods)
                    
                    if new_score < current_score:
                        current_edges.remove(e_in_idx)
                        current_edges.add(e_out_idx)
                        
                        current_distinct_sums = new_distinct_sums
                        current_distinct_prods = new_distinct_prods
                        current_score = new_score
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = [all_edges[idx] for idx in current_edges]
                        
                        break
                if improved:
                    break
            
        if current_score < best_score:
            best_score = current_score
            best_edges = [all_edges[idx] for idx in current_edges]
            
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit (MILP needs more time)
    
    best_raw = float('inf')
    best_result = None
    
    # Candidate value sets
    # Small integers are best for collisions in both sums and products
    value_sets = [
        list(range(1, 15)),
        list(range(2, 16)),
        list(range(3, 17)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48], # Multiplicative structure
        [1, 2, 3, 5, 7, 10, 14, 15, 20, 21, 28, 30, 35, 42], # Divisors of 42/210
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (bad for sums, good for prods)
    ]
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
            
        edges, raw = _mip_solve_edge_selection(vals, 35, deadline)
        
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': vals, 'edges': edges}
            
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
        
    # Format output
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
