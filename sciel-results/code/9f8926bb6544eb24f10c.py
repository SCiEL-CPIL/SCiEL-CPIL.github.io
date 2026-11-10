import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 14
    m = 35
    
    # Define candidate graphs
    # Circulant graphs on 14 nodes with steps S.
    # Edges: {(i, (i+k)%14) for i in range(14) for k in S}
    # Note: We must ensure no duplicates and exactly 35 edges.
    # Step 7 adds 7 edges (matching).
    # Steps 1-6 add 14 edges each.
    # So 2 steps from {1..6} + step 7 = 14 + 14 + 7 = 35 edges.
    
    candidate_graphs = []
    for s1 in range(1, 7):
        for s2 in range(s1 + 1, 7):
            steps = [s1, s2, 7]
            edges = set()
            for i in range(n):
                for k in steps:
                    j = (i + k) % n
                    if i < j:
                        edges.add((i, j))
                    else:
                        edges.add((j, i))
            # Verify edge count
            if len(edges) == 35:
                candidate_graphs.append(list(edges))
    
    # If no graphs found (shouldn't happen), fallback to a simple one
    if not candidate_graphs:
        # Fallback: First 35 pairs
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
                if len(edges) == 35:
                    break
            if len(edges) == 35:
                break
        candidate_graphs.append(edges)
        
    # Define candidate value sets
    candidate_values = []
    
    # 1. Small integers
    candidate_values.append(list(range(1, 15)))
    candidate_values.append(list(range(2, 16)))
    candidate_values.append(list(range(1, 14)) + [15]) # Just in case
    
    # 2. Arithmetic Progressions
    for start in range(1, 10):
        for step in [1, 2, 3, 4]:
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidate_values.append(vals)
                
    # 3. Random small integers
    random.seed(42)
    for _ in range(20):
        vals = random.sample(range(1, 100), n)
        candidate_values.append(vals)
        
    # 4. Random integers with small factors (heuristic for product collisions)
    # Generate numbers that are products of small primes
    small_primes = [2, 3, 5, 7, 11]
    for _ in range(20):
        vals = set()
        while len(vals) < n:
            # Pick random exponents
            v = 1
            for p in small_primes:
                v *= p ** random.randint(0, 3)
            if v > 10000:
                continue
            vals.add(v)
        if len(vals) == n:
            candidate_values.append(list(vals))

    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
        
    # Local Search / Simulated Annealing
    def local_search(edges, initial_values, time_limit=2.0):
        current_values = initial_values[:]
        current_raw = evaluate(current_values, edges)
        best_values_local = current_values[:]
        best_raw_local = current_raw
        
        # Precompute edge list for fast access
        edge_list = edges
        
        t_start = time.time()
        temp = 10.0
        cooling = 0.99
        
        while time.time() - t_start < time_limit:
            # Pick a random move: swap two values
            i, j = random.sample(range(n), 2)
            
            # Calculate delta
            # This is expensive if done naively. 
            # We can optimize by only checking edges incident to i and j.
            
            # For now, naive evaluation is O(m) = 35 ops. 
            # 35 ops is very fast. We can do many iterations.
            
            new_values = current_values[:]
            new_values[i], new_values[j] = current_values[j], new_values[i]
            
            new_raw = evaluate(new_values, edge_list)
            
            delta = new_raw - current_raw
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_values = new_values
                current_raw = new_raw
                
                if current_raw < best_raw_local:
                    best_raw_local = current_raw
                    best_values_local = current_values[:]
                    
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
                
        return best_raw_local, best_values_local

    # Iterate over candidates
    # To manage time, we assign a small time slice to each (graph, values) pair
    total_pairs = len(candidate_graphs) * len(candidate_values)
    time_per_pair = 40.0 / max(1, total_pairs)
    if time_per_pair > 1.0:
        time_per_pair = 1.0
        
    for edges in candidate_graphs:
        if time.time() > deadline:
            break
        for vals in candidate_values:
            if time.time() > deadline:
                break
            
            # Ensure values are valid
            if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
                continue
                
            raw, permuted_vals = local_search(edges, vals, time_limit=time_per_pair)
            
            if raw < best_raw:
                best_raw = raw
                best_values = permuted_vals
                best_edges = [list(e) for e in edges]
                
            # Early termination if we reach a very good score
            # Theoretical minimum is likely around 10-15.
            if best_raw <= 12:
                # We can try to squeeze more, but 12 is very good.
                # Let's just break to save time for other candidates?
                # No, let's let it run a bit more or break if very low.
                pass

    # Fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in candidate_graphs[0]]

    # Final validation and type conversion
    return {
        'values': [int(v) for v in best_values],
        'edges': [list(e) for e in best_edges]
    }

# EVOLVE-BLOCK-END
