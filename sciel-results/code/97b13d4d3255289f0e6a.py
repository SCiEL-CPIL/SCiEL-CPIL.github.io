"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _count_distinct(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return len(sums), len(prods)

def _optimize_values(initial_values, max_iters=2000, seed=42):
    """
    Local search to find 14 distinct integers <= 10000 that minimize
    the number of distinct sums and products over all possible pairs.
    """
    random.seed(seed)
    values = initial_values[:]
    n = len(values)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_cost(vals):
        sums = set()
        prods = set()
        for i, j in all_pairs:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    current_cost = get_cost(values)
    best_values = values[:]
    best_cost = current_cost
    temp = 10.0
    
    for t in range(max_iters):
        # Cooling schedule
        temp = max(0.1, 10.0 * (1.0 - t / max_iters))
        
        # Pick a random index to change
        idx = random.randint(0, n - 1)
        old_val = values[idx]
        
        # Generate a new value
        # Strategy: try small numbers often, but allow large jumps
        if random.random() < 0.7:
            new_val = random.randint(1, 50)
        else:
            new_val = random.randint(1, 10000)
            
        # Ensure distinctness and positivity
        if new_val == old_val:
            continue
        if any(v == new_val for v in values if v != old_val):
            continue
        if new_val <= 0 or new_val > 10000:
            continue
            
        values[idx] = new_val
        new_cost = get_cost(values)
        
        # Acceptance criterion (Simulated Annealing)
        if new_cost < best_cost:
            best_values = values[:]
            best_cost = new_cost
            current_cost = new_cost
        else:
            # Probabilistic acceptance of worse states
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_cost = new_cost
            else:
                values[idx] = old_val
                
    return best_values, best_cost

def _optimize_edges(values, max_iters=5000, seed=42):
    """
    Local search to select 35 edges that minimize the max distinct sums/products.
    """
    random.seed(seed)
    n = len(values)
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = 35
    
    # Initialize with a random set of 35 edges
    current_edges = set(random.sample(all_possible_edges, m))
    
    def get_edge_cost(edge_set):
        sums = set()
        prods = set()
        for i, j in edge_set:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
        
    current_cost = get_edge_cost(current_edges)
    best_edges = frozenset(current_edges)
    best_cost = current_cost
    temp = 5.0
    
    for t in range(max_iters):
        temp = max(0.5, 5.0 * (1.0 - t / max_iters))
        
        # Move: Remove one random edge, add one random new edge
        edge_to_remove = random.choice(list(current_edges))
        current_edges.remove(edge_to_remove)
        
        # Find a random edge not in current_edges
        # Optimization: Pre-generate a list of all edges and pick one not in set
        # Since m=35 and total=91, most edges are available.
        # Simple approach: pick random until we find one not in set
        attempts = 0
        while attempts < 10:
            candidate_edge = random.choice(all_possible_edges)
            if candidate_edge not in current_edges:
                break
            attempts += 1
        else:
            # Failed to find a candidate (should not happen with 91 edges)
            current_edges.add(edge_to_remove)
            continue
            
        current_edges.add(candidate_edge)
        
        new_cost = get_edge_cost(current_edges)
        
        if new_cost < best_cost:
            best_edges = frozenset(current_edges)
            best_cost = new_cost
            current_cost = new_cost
        else:
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_cost = new_cost
            else:
                # Reject move: revert
                current_edges.remove(candidate_edge)
                current_edges.add(edge_to_remove)
                
    return list(best_edges), best_cost

def construct():
    start_time = time.time()
    limit = 55.0 # Leave 5s buffer
    
    # Initial candidate value sets
    inits = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (bad for sums, but good for products? No, products are unique too)
    ]
    
    best_overall_cost = float('inf')
    best_vals = None
    best_edges = None
    
    # Phase 1: Optimize Values
    # We try to find a value set with low global distinct counts.
    # We limit time for this phase.
    time_spent = 0
    for init in inits:
        if time.time() - start_time > limit * 0.5:
            break
        # Run value optimization
        # Use different seeds for diversity
        seed = hash(tuple(init)) % 10000
        vals, cost = _optimize_values(init, max_iters=1000, seed=seed)
        
        # Phase 2: Optimize Edges for this value set
        if time.time() - start_time > limit * 0.8:
            break
            
        edges, edge_cost = _optimize_edges(vals, max_iters=2000, seed=seed + 1)
        
        if edge_cost < best_overall_cost:
            best_overall_cost = edge_cost
            best_vals = vals
            best_edges = edges
            
    # Fallback
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges, _ = _optimize_edges(best_vals, max_iters=100, seed=1)
        
    # Ensure types are correct
    best_vals = [int(v) for v in best_vals]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_vals, 'edges': best_edges}
# EVOLVE-BLOCK-END
