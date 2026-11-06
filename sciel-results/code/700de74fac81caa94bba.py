"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_makespan(pt, order):
    """
    Calculate makespan for a given order.
    Uses cumulative sums and broadcasting for speed.
    pt: (n_jobs, n_machines)
    order: array of job indices
    Returns: makespan (scalar)
    """
    n_jobs, n_machines = pt.shape
    # Reorder processing times according to the sequence
    # shape: (n_jobs, n_machines)
    p = pt[order]
    
    # C[i, j] = completion time of job i on machine j
    # C[i, 0] = C[i-1, 0] + p[i, 0]
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
    
    # Initialize C
    # C[0, j] = C[0, j-1] + p[0, j]
    # C[i, 0] = C[i-1, 0] + p[i, 0]
    
    # We can compute this iteratively but vectorize the inner loop if possible?
    # For n_jobs < 1000, a simple loop is often faster than complex vectorization due to overhead.
    # However, we can use a trick:
    # The makespan is the sum of the critical path.
    
    # Let's use a standard loop which is robust and reasonably fast for moderate N.
    # Optimization: use local variables for speed
    c = 0.0
    # First machine
    for i in range(n_jobs):
        c += p[i, 0]
        # For subsequent machines
        for j in range(1, n_machines):
            # We need C[i-1, j] and C[i, j-1]
            # This simple loop is O(N*M). 
            # To make it faster, we can maintain an array 'prev_mach' representing C[i-1, :]
            # and 'curr_mach' representing C[i, :]
            # Actually, let's just implement the DP table properly but efficiently.
            pass
            
    # Better approach for speed:
    # C[i][j] depends on C[i-1][j] and C[i][j-1].
    # Let's maintain an array 'last' of size n_machines where last[j] = C[i-1][j]
    # And compute curr[j] for the current job i.
    
    last = np.zeros(n_machines)
    for i in range(n_jobs):
        curr = np.zeros(n_machines)
        curr[0] = last[0] + p[i, 0]
        for j in range(1, n_machines):
            curr[j] = max(last[j], curr[j-1]) + p[i, j]
        last = curr
        
    return last[-1]

def _neighborhood(order, pt, n_neighbors=10):
    """
    Generate neighbors using insertion moves (remove job, insert at different position).
    This is generally more effective than swaps for flow-shop.
    """
    n = len(order)
    neighbors = []
    # Sample random pairs (i, j) for removal and insertion
    for _ in range(n_neighbors):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        new_order = order.copy()
        job = new_order[i]
        # Remove job at i
        del new_order[i]
        # Insert at j (adjust index if i < j)
        # If we remove at i, the indices after i shift left.
        # We want to insert into the list *after* removal.
        # If j > i, the target index in the new list is j-1? No.
        # Let's just pop and insert.
        # new_order is already popped.
        # Insert at position j. If j > i, the position j in original corresponds to j-1 in new?
        # Actually, `list.insert(idx, item)` inserts before idx.
        # If we remove at i, and want to place it at original index j:
        # If j > i, we insert at j-1? No, if we remove item at i, the item at j becomes j-1.
        # So if we want it to end up at position j in the new list (0-indexed), we insert at j.
        # But we are sampling random positions. Let's just pick a random insertion point in the new list.
        ins_pos = random.randint(0, n - 1)
        new_order.insert(ins_pos, job)
        neighbors.append(new_order)
        
    return neighbors

def order_jobs(processing_times):
    """
    Evolutionary Strategy for Flow Shop Scheduling.
    1. Initialize population with NEH heuristic variations.
    2. Iteratively select best, generate neighbors via insertion, evaluate, update.
    3. Stop before time limit.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # --- Initialization: NEH Heuristic ---
    # Sort by total processing time descending
    total_times = processing_times.sum(axis=1)
    order_init = np.argsort(-total_times)
    
    # Construct NEH sequence
    seq = [order_init[0]]
    for idx in order_init[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [idx] + seq[pos:]
            ms = _fast_makespan(processing_times, np.array(trial))
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, idx)
        
    best_order = np.array(seq)
    best_ms = _fast_makespan(processing_times, best_order)
    
    # Population size
    pop_size = 10
    # Create initial population: Best + random perturbations of best
    population = [best_order.copy()]
    for _ in range(pop_size - 1):
        perturbed = best_order.copy()
        # Random swap
        i, j = random.sample(range(n_jobs), 2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        population.append(perturbed)
        
    # Evaluate initial population
    pop_ms = [_fast_makespan(processing_times, p) for p in population]
    
    # Main Loop
    while time.time() - start_time < time_limit:
        # Select best individual
        best_idx = np.argmin(pop_ms)
        current_best_order = population[best_idx]
        current_best_ms = pop_ms[best_idx]
        
        # Generate offspring
        n_offspring = 5
        offspring = []
        for _ in range(n_offspring):
            # Use insertion neighborhood on the current best
            # Pick a random job to move
            i = random.randint(0, n_jobs - 1)
            job = current_best_order[i]
            # Remove
            new_order = np.delete(current_best_order, i)
            # Insert at random position
            j = random.randint(0, n_jobs - 1)
            new_order = np.insert(new_order, j, job)
            offspring.append(new_order)
            
        # Evaluate offspring
        off_ms = [_fast_makespan(processing_times, o) for o in offspring]
        
        # Combine population and offspring
        all_orders = population + offspring
        all_ms = pop_ms + off_ms
        
        # Select next population (keep best 'pop_size' individuals)
        # Sort by makespan
        indices = np.argsort(all_ms)
        next_pop_indices = indices[:pop_size]
        
        new_population = [all_orders[i].copy() for i in next_pop_indices]
        new_pop_ms = [all_ms[i] for i in next_pop_indices]
        
        population = new_population
        pop_ms = new_pop_ms
        
        # If we found a new global best, we could restart or intensify, 
        # but simple selection usually works well for ES.
        
        # Check if we should break early if no improvement for many generations?
        # Not strictly necessary with time limit, but helps if stuck.
        
    # Return best order
    return list(population[np.argmin(pop_ms)])
# EVOLVE-BLOCK-END
