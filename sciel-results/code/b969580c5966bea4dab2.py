"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter
import math

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    """Compute the objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    # All possible edges
    all_edges = list(combinations(range(n), 2))
    edge_set = set(all_edges)
    
    # Initial value sets to try as seeds for local search
    # Small integers are good for keeping products small and distinct count low.
    # Symmetric sets might help with sums.
    base_sets = [
        list(range(1, n + 1)),
        list(range(2, 2 * n + 1, 2)), # Evens
        [i * 3 for i in range(1, n + 1)],
        [10 * i for i in range(1, n + 1)],
        [i * 11 for i in range(1, n + 1)],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43], # Primes (bad for products? unique)
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (unique products)
    ]
    
    # Add some random small sets
    for _ in range(20):
        base_sets.append(random.sample(range(1, 50), n))
        
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    def evaluate_state(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def local_search(values, edges, time_limit):
        """
        Perform Simulated Annealing on the joint state of values and edges.
        """
        current_values = list(values)
        current_edges = [list(e) for e in edges]
        current_raw = evaluate_state(current_values, current_edges)
        
        best_values_ls = list(current_values)
        best_edges_ls = [list(e) for e in current_edges]
        best_raw_ls = current_raw
        
        # Precompute adjacency or just keep a set of edges for O(1) lookup
        current_edge_set = set(tuple(e) for e in current_edges)
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.999
        
        steps = 0
        while time.time() < time_limit and T > T_min:
            # Random Move
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap two values
                i, j = random.sample(range(n), 2)
                current_values[i], current_values[j] = current_values[j], current_values[i]
                new_raw = evaluate_state(current_values, current_edges)
                # Revert if not accepted? No, SA accepts based on delta.
                delta = new_raw - current_raw
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_raw = new_raw
                    if current_raw < best_raw_ls:
                        best_raw_ls = current_raw
                        best_values_ls = list(current_values)
                        best_edges_ls = [list(e) for e in current_edges]
                else:
                    current_values[i], current_values[j] = current_values[j], current_values[i]
                    
            else:
                # Swap an edge with a non-edge
                # Pick a random existing edge
                e_idx = random.randint(0, m - 1)
                e_old = tuple(current_edges[e_idx])
                
                # Pick a random non-edge
                # Efficient way: pick random pair, if it's in edge_set, retry
                # Or precompute list of non-edges? 
                # 91 total, 35 used. 56 available.
                # Just pick random pair until valid
                while True:
                    i, j = random.sample(range(n), 2)
                    e_new = (i, j)
                    if e_new not in current_edge_set:
                        break
                        
                # Apply move
                current_edges[e_idx] = list(e_new)
                current_edge_set.remove(e_old)
                current_edge_set.add(e_new)
                
                new_raw = evaluate_state(current_values, current_edges)
                delta = new_raw - current_raw
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_raw = new_raw
                    if current_raw < best_raw_ls:
                        best_raw_ls = current_raw
                        best_values_ls = list(current_values)
                        best_edges_ls = [list(e) for e in current_edges]
                else:
                    # Revert
                    current_edges[e_idx] = list(e_old)
                    current_edge_set.remove(e_new)
                    current_edge_set.add(e_old)
                    
            T *= cooling_rate
            steps += 1
            
        return best_values_ls, best_edges_ls, best_raw_ls

    # Run local search for each base set
    # Allocate time budget per base set
    num_bases = len(base_sets)
    time_per_base = (deadline - start_time) / num_bases
    
    for idx, base_vals in enumerate(base_sets):
        if time.time() > deadline - 5:
            break
            
        # Ensure distinct and valid
        if len(set(base_vals)) != n or any(v < 1 or v > maximum for v in base_vals):
            continue
            
        # Initial edge selection: Random 35 edges
        init_edges = random.sample(all_edges, m)
        
        # Run local search
        t_start = time.time()
        t_limit = min(t_start + time_per_base, deadline - 5)
        
        vals, edges, raw = local_search(base_vals, init_edges, t_limit)
        
        if raw < best_raw:
            best_raw = raw
            best_values = vals
            best_edges = edges
            
        # Optional: If we find a very good solution, we can restart from it with a different base
        # But for now, just iterate through bases.

    # Final polish: If we have time, run SA on the best found solution again with higher T
    if best_values is not None and time.time() < deadline - 2:
        vals, edges, raw = local_search(best_values, best_edges, deadline - 1)
        if raw < best_raw:
            best_raw = raw
            best_values = vals
            best_edges = edges

    # Fallback
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END
