"""Strategy: Joint Simulated Annealing on (values, edges) to maximize additive and multiplicative collisions simultaneously."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    
    # Helper to check validity of a state
    def is_valid(vals, edges):
        if len(vals) != n:
            return False
        if len(edges) != m:
            return False
        if len(set(vals)) != n:
            return False
        if any(v < 1 or v > 10000 for v in vals):
            return False
        seen_edges = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen_edges:
                return False
            seen_edges.add(key)
        return True

    def evaluate(vals, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))

    # Initial State: Small integers 1..14 often yield good collisions.
    # Edges: Random 35 edges.
    initial_vals = list(range(1, 15))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # To make initial edges "good", we can pick edges that have small sums/prods?
    # Or just random. Let's try random first, SA will fix it.
    initial_edges = [list(e) for e in random.sample(all_pairs, m)]
    
    current_vals = initial_vals[:]
    current_edges = initial_edges[:]
    current_cost = evaluate(current_vals, current_edges)
    
    best_vals = current_vals[:]
    best_edges = current_edges[:]
    best_cost = current_cost
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    step = 0
    while time.time() < deadline - 2.0:
        step += 1
        
        # 1. Generate Neighbor
        new_vals = current_vals[:]
        new_edges = current_edges[:]
        
        # Randomly choose between value perturbation and edge perturbation
        # Or do both?
        # Let's do a mix.
        
        r = random.random()
        
        if r < 0.4:
            # Perturb Values: Swap two values or change one value
            if random.random() < 0.5:
                i, j = random.sample(range(n), 2)
                new_vals[i], new_vals[j] = new_vals[j], new_vals[i]
            else:
                i = random.randint(0, n-1)
                # Try to move value towards smaller numbers to increase collision chance
                # Or random step
                delta = random.randint(-5, 5)
                new_val = new_vals[i] + delta
                if 1 <= new_val <= 10000 and new_val not in new_vals:
                    new_vals[i] = new_val
                else:
                    # If invalid, try a random small value
                    new_val = random.randint(1, 20)
                    if new_val not in new_vals:
                        new_vals[i] = new_val
                    else:
                        # Skip this step if we can't make a valid value change easily
                        continue 
        elif r < 0.8:
            # Perturb Edges: Swap an edge in with an edge out
            i = random.randint(0, m-1)
            old_edge = new_edges[i]
            
            # Find a candidate edge not in current set
            # To be efficient, we can just pick a random pair and check if it's already there
            # Since m=35 and total=91, probability of collision is ~38%.
            # Retry a few times.
            new_edge = None
            for _ in range(10):
                candidate = random.choice(all_pairs)
                if candidate not in new_edges:
                    new_edge = list(candidate)
                    break
            
            if new_edge is None:
                continue
                
            new_edges[i] = new_edge
        else:
            # Perturb both
            # Value swap
            i, j = random.sample(range(n), 2)
            new_vals[i], new_vals[j] = new_vals[j], new_vals[i]
            
            # Edge swap
            k = random.randint(0, m-1)
            for _ in range(10):
                candidate = random.choice(all_pairs)
                if candidate not in new_edges:
                    new_edges[k] = list(candidate)
                    break
            else:
                continue

        # Check validity (distinctness of values)
        if len(set(new_vals)) != n:
            continue
            
        new_cost = evaluate(new_vals, new_edges)
        
        # Acceptance
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or (T > T_min and random.random() < math.exp(-delta_cost / T)):
            current_vals = new_vals
            current_edges = new_edges
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_vals = current_vals[:]
                best_edges = current_edges[:]
                
        T *= cooling
        
        # If T is too low, reheat slightly to escape local minima?
        # Or just let it cool.
        
    # Final cleanup and return
    # Ensure types are Python int
    final_vals = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Sort edges for determinism (optional but good)
    final_edges.sort()
    
    return {
        'values': final_vals,
        'edges': final_edges
    }
# EVOLVE-BLOCK-END
