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

def _generate_initial_state():
    """
    Generate a random initial state:
    - 14 distinct positive integers <= 10000
    - 35 unique edges among 14 nodes
    """
    # Values: Small numbers tend to have more collisions.
    # Pick from 1..50 to ensure high collision probability initially.
    values = random.sample(range(1, 51), 14)
    random.shuffle(values)
    
    # Edges: Random 35 edges from C(14,2)=91
    all_edges = list(combinations(range(14), 2))
    random.shuffle(all_edges)
    edges = all_edges[:35]
    
    return values, edges

def _joint_simulated_annealing(time_limit):
    """
    Jointly optimize values and edges using Simulated Annealing.
    
    State: (values, edges)
    Moves:
    1. Perturb one value (replace with a random distinct valid value).
    2. Swap one edge (remove one, add another).
    3. Simultaneously perturb one value and swap one edge (higher probability of escape).
    
    Cost: max(distinct_sums, distinct_products)
    """
    start_time = time.time()
    
    # Initial state
    values, edges = _generate_initial_state()
    
    # Precompute all possible edges for quick lookup
    all_edges = list(combinations(range(14), 2))
    edge_set = set(edges)
    
    def calc_cost(vals, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            su = vals[u]
            sv = vals[v]
            sums.add(su + sv)
            prods.add(su * sv)
        return max(len(sums), len(prods))
    
    current_cost = calc_cost(values, edges)
    best_values = list(values)
    best_edges = list(edges)
    best_cost = current_cost
    
    # SA Parameters
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    # Helper to get a random valid new value
    def get_new_value(current_vals):
        # Bias towards smaller numbers for collision
        # 80% chance to pick from 1..50, 20% from 1..100
        if random.random() < 0.8:
            upper = 50
        else:
            upper = 100
        
        # Ensure distinctness
        while True:
            new_v = random.randint(1, upper)
            if new_v not in current_vals:
                return new_v
    
    # Helper to get a random edge not in current set
    def get_new_edge(current_edge_set):
        # To avoid O(N) scan every time, we can just pick random pairs and check
        # But C(14,2)=91 is small, so scanning is fine.
        # Or keep a list of all edges and pick a random one, checking if it's in set.
        while True:
            e = random.choice(all_edges)
            if e not in current_edge_set:
                return e
    
    # Main SA Loop
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Choose move type
        move_type = random.random()
        old_vals = list(values)
        old_edges = list(edges)
        old_edge_set = set(edges)
        
        if move_type < 0.4:
            # 40%: Change one value
            idx = random.randint(0, 13)
            new_val = get_new_value(old_vals)
            new_vals = old_vals
            new_vals[idx] = new_val
            new_edges = old_edges
            new_edge_set = old_edge_set
        elif move_type < 0.8:
            # 40%: Swap one edge
            idx = random.randint(0, 34)
            out_edge = old_edges[idx]
            in_edge = get_new_edge(old_edge_set)
            
            new_edges = list(old_edges)
            new_edges[idx] = in_edge
            new_vals = old_vals
            new_edge_set = set(new_edges)
        else:
            # 20%: Change one value AND swap one edge
            idx_v = random.randint(0, 13)
            new_val = get_new_value(old_vals)
            new_vals = list(old_vals)
            new_vals[idx_v] = new_val
            
            idx_e = random.randint(0, 34)
            out_edge = old_edges[idx_e]
            in_edge = get_new_edge(old_edge_set)
            new_edges = list(old_edges)
            new_edges[idx_e] = in_edge
            new_edge_set = set(new_edges)
        
        new_cost = calc_cost(new_vals, new_edges)
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            values = new_vals
            edges = new_edges
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_values = list(values)
                best_edges = list(edges)
        
        T *= cooling
        
    return best_values, best_edges, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_global_score = float('inf')
    best_global_result = None
    
    # Run multiple independent SA runs to avoid getting stuck in local minima
    # Each run takes ~5s. We can afford ~10 runs.
    num_runs = 10
    
    for i in range(num_runs):
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
        
        # Allocate remaining time evenly to remaining runs, with a cap per run
        # to ensure diversity.
        time_per_run = min(5.0, time_left / (num_runs - i))
        
        values, edges, score = _joint_simulated_annealing(time_per_run)
        
        # Validate constraints strictly
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        if len(edges) != 35:
            continue
        # Check for duplicate edges
        if len(set(map(tuple, edges))) != 35:
            continue
        # Check indices
        if any(u >= 14 or v >= 14 or u == v for u, v in edges):
            continue
            
        if score < best_global_score:
            best_global_score = score
            best_global_result = {
                'values': [int(v) for v in values],
                'edges': [list(e) for e in edges]
            }
            
        # Early exit if we find a very good solution
        if best_global_score <= 10:
            break
            
    if best_global_result is None:
        # Fallback: simple heuristic
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {
            'values': values,
            'edges': edges
        }
        
    return best_global_result
