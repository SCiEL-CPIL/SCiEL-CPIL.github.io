"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _select_edges(values):
    """
    Selects 35 edges to minimize max(distinct_sums, distinct_products).
    Uses a greedy approach with tie-breaking.
    """
    n = len(values)
    m = 35
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
        
        # Shuffle available to break ties randomly, encouraging diversity in SA
        random.shuffle(available)
        
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
                # Prefer edges that add fewer new items
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
        
    return selected_edges, max(len(current_sums), len(current_prods))

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    best_values = None
    best_score = float('inf')
    best_edges = None
    
    # 1. Initialize with a diverse set of candidates
    candidates = [
        list(range(1, n + 1)),
        list(range(2, 2*n+2, 2)), # Even numbers
        list(range(1, 2*n+1, 2)), # Odd numbers
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], # Primes
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # Will be fixed by distinctness check
    ]
    
    for c in candidates:
        # Make distinct
        if len(set(c)) != n:
            c = list(range(1, n+1))
        if any(v < 1 or v > 10000 for v in c):
            continue
        edges, score = _select_edges(c)
        if score < best_score:
            best_score = score
            best_values = c[:]
            best_edges = edges[:]
            
    # 2. Simulated Annealing
    current_values = best_values[:]
    current_score = best_score
    current_edges = best_edges[:]
    
    T = 50.0
    cooling = 0.995
    min_T = 1.0
    
    # Track stagnation
    last_improvement = time.time()
    stagnation_limit = 5.0
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        idx = random.randint(0, n - 1)
        old_val = current_values[idx]
        
        # Move generation: 
        # 10% chance to pick a random value from 1..10000
        # 90% chance to pick a value near current or in a small range
        if random.random() < 0.1:
            new_val = random.randint(1, 10000)
        else:
            # Bias towards small numbers as they collide more
            new_val = random.randint(1, 50)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_values:
            continue
            
        current_values[idx] = new_val
        new_edges, new_score = _select_edges(current_values)
        
        if new_score < current_score:
            current_score = new_score
            current_edges = new_edges
            if new_score < best_score:
                best_score = new_score
                best_values = current_values[:]
                best_edges = current_edges[:]
                last_improvement = time.time()
        else:
            delta = new_score - current_score
            # Accept worse moves to escape local minima
            if random.random() < pow(2.71828, -delta / T):
                current_score = new_score
                current_edges = new_edges
            else:
                current_values[idx] = old_val
                
        T *= cooling
        
        # Restart if stuck for too long
        if time.time() - last_improvement > stagnation_limit:
            # Reset to best known and increase T slightly? Or just continue.
            # Let's just let it cool down.
            pass

    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, _ = _select_edges(best_values)
        
    # Ensure types are Python int and list
    return {'values': [int(v) for v in best_values], 'edges': [[int(i), int(j)] for i, j in best_edges]}
# EVOLVE-BLOCK-END
