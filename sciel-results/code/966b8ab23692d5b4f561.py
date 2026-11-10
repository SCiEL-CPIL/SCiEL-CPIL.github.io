import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # We aim for the maximum n. Known valid n is 34.
    # We will try n=34, then 33, etc.
    
    # Precompute K4 and K5 structures for each n we might try
    # But since n changes, we do it inside solve_hypergraph_ramsey
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
            
        # Allocate time for this n
        # Give more time to higher n
        time_limit = 25.0
        if n < 34:
            time_limit = 15.0
        if n < 30:
            time_limit = 8.0
        if n < 25:
            time_limit = 3.0
        if n < 20:
            time_limit = 1.0
            
        # Adjust time limit to not exceed deadline
        remaining = deadline - time.time()
        if time_limit > remaining:
            time_limit = remaining
            if time_limit < 0.5:
                break

        solution = solve_hypergraph_ramsey(n, time_limit)
        if solution is not None:
            return solution

    # Fallback: Return a small valid construction if all else fails
    if time.time() < deadline - 1.0:
        solution = solve_hypergraph_ramsey(10, 1.0)
        if solution:
            return solution
            
    # Absolute fallback: n=3 with no triples
    triples = list(combinations(range(3), 3))
    return {'n': 3, 'colors': [0] * len(triples)}

def solve_hypergraph_ramsey(n, time_limit):
    # Precomputation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Build K4s and K5s
    k4s = []
    k4_map = [[] for _ in range(num_triples)]
    for quad in combinations(range(n), 4):
        edges = []
        for tri in combinations(quad, 3):
            idx = triple_index[tri]
            edges.append(idx)
        k4_idx = len(k4s)
        k4s.append(edges)
        for e in edges:
            k4_map[e].append(k4_idx)
            
    k5s = []
    k5_map = [[] for _ in range(num_triples)]
    for quint in combinations(range(n), 5):
        edges = []
        for tri in combinations(quint, 3):
            idx = triple_index[tri]
            edges.append(idx)
        k5_idx = len(k5s)
        k5s.append(edges)
        for e in edges:
            k5_map[e].append(k5_idx)
            
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    # Multiple restarts strategy
    # Each restart gets a fraction of the time
    # We'll do up to 10 restarts, each with time_limit/10
    # But we'll adapt: if we find a solution, return immediately
    
    best_solution = None
    best_violations = float('inf')
    
    # Number of restarts
    max_restarts = 8
    time_per_restart = time_limit / max_restarts
    
    # But for the first few restarts, give more time
    # Actually, let's just do a single long run with periodic restarts within it
    
    # Better: run local search, and if stuck (no improvement for X steps), restart
    # Let's implement a more sophisticated local search with simulated annealing
    
    # State
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Violation Counts
    red_k4_counts = [0] * num_k4
    blue_k5_counts = [0] * num_k5
    
    total_violations = 0
    
    # Initialize counts
    for i, edges in enumerate(k4s):
        count = 0
        for e in edges:
            if colors[e] == 0:
                count += 1
        red_k4_counts[i] = count
        if count == 4:
            total_violations += 1
            
    for i, edges in enumerate(k5s):
        count = 0
        for e in edges:
            if colors[e] == 1:
                count += 1
        blue_k5_counts[i] = count
        if count == 10:
            total_violations += 1
            
    best_colors = colors[:]
    best_violations = total_violations
    
    if total_violations == 0:
        return {'n': n, 'colors': colors}
        
    start = time.time()
    T = 2.0
    cooling = 0.9998
    min_T = 1e-4
    
    # Track steps without improvement
    steps_since_improvement = 0
    max_steps_no_improvement = 5000
    
    # Precompute which triples are in which K4s and K5s for fast access
    # Already done with k4_map and k5_map
    
    while time.time() - start < time_limit:
        # Pick a random triple to flip
        idx = random.randint(0, num_triples - 1)
        old_val = colors[idx]
        new_val = 1 - old_val
        
        delta_violations = 0
        
        # Update K4 counts
        for k4_idx in k4_map[idx]:
            old_count = red_k4_counts[k4_idx]
            if old_val == 0:
                new_count = old_count - 1
            else:
                new_count = old_count + 1
            
            if old_count == 4:
                delta_violations -= 1
            if new_count == 4:
                delta_violations += 1
                
            red_k4_counts[k4_idx] = new_count
            
        # Update K5 counts
        for k5_idx in k5_map[idx]:
            old_count = blue_k5_counts[k5_idx]
            if old_val == 1:
                new_count = old_count - 1
            else:
                new_count = old_count + 1
                
            if old_count == 10:
                delta_violations -= 1
            if new_count == 10:
                delta_violations += 1
                
            blue_k5_counts[k5_idx] = new_count
            
        new_violations = total_violations + delta_violations
        
        # Accept or Reject
        if new_violations < total_violations:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-delta_violations / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            colors[idx] = new_val
            total_violations = new_violations
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if total_violations == 0:
                return {'n': n, 'colors': colors}
        else:
            # Revert changes
            colors[idx] = old_val
            for k4_idx in k4_map[idx]:
                if old_val == 0:
                    red_k4_counts[k4_idx] += 1
                else:
                    red_k4_counts[k4_idx] -= 1
            for k5_idx in k5_map[idx]:
                if old_val == 1:
                    blue_k5_counts[k5_idx] += 1
                else:
                    blue_k5_counts[k5_idx] -= 1
            steps_since_improvement += 1
            
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Restart if stuck
        if steps_since_improvement > max_steps_no_improvement:
            # Restart with random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            total_violations = 0
            for i, edges in enumerate(k4s):
                count = 0
                for e in edges:
                    if colors[e] == 0:
                        count += 1
                red_k4_counts[i] = count
                if count == 4:
                    total_violations += 1
            for i, edges in enumerate(k5s):
                count = 0
                for e in edges:
                    if colors[e] == 1:
                        count += 1
                blue_k5_counts[i] = count
                if count == 10:
                    total_violations += 1
            steps_since_improvement = 0
            T = 2.0
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors[:]
                
            if total_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Check time
        if time.time() - start > time_limit:
            break
        
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None
