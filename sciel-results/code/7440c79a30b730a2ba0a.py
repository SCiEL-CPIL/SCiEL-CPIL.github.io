"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_vec(perm, pt):
    """Vectorized makespan computation using the flow-shop recurrence.
    
    The recurrence is: C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    This is computed in O(n*m) with minimal Python overhead.
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    
    # pt_perm[i] = processing times for job at position i
    pt_perm = pt[perm]  # shape (n, m)
    
    # C[i][j] = completion time of job i on machine j
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + pt_perm[i][j]
    # We can compute this iteratively, but vectorize over machines for each job
    
    # Use a 2D array: C[i][j]
    C = np.zeros((n, m), dtype=float)
    C[0] = np.cumsum(pt_perm[0])
    for i in range(1, n):
        C[i, 0] = C[i-1, 0] + pt_perm[i, 0]
        C[i, 1:] = np.maximum(C[i-1, 1:], C[i, :-1]) + pt_perm[i, 1:]
    
    return C[-1, -1]

def _neh(processing_times):
    """NEH heuristic for flow shop scheduling using vectorized makespan."""
    n, m = processing_times.shape
    total_pt = processing_times.sum(axis=1)
    order = np.argsort(-total_pt)
    
    # Build permutation by inserting jobs one by one in best position
    perm = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_vec(np.array(trial), processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=int)

def _insertion_ls(perm, pt, max_passes=3):
    """Fast insertion local search: for each job, try all insertion positions."""
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    best_perm = perm.copy()
    best_ms = _makespan_vec(best_perm, pt)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            job = best_perm[i]
            # Remove job from position i
            remaining = np.concatenate([best_perm[:i], best_perm[i+1:]])
            # Try all insertion positions
            best_pos = i
            best_trial_ms = float('inf')
            for pos in range(n - 1):
                trial = np.concatenate([remaining[:pos], [job], remaining[pos:]])
                ms = _makespan_vec(trial, pt)
                if ms < best_trial_ms:
                    best_trial_ms = ms
                    best_pos = pos
            
            # Reconstruct with best position
            new_perm = np.concatenate([remaining[:best_pos], [job], remaining[best_pos:]])
            if best_trial_ms < best_ms - 1e-9:
                best_ms = best_trial_ms
                best_perm = new_perm
                improved = True
        
        if not improved:
            break
    
    return best_perm

def _order_crossover(p1, p2, rng):
    """Order crossover (OX) for permutations."""
    n = len(p1)
    if n < 3:
        return p1.copy()
    
    # Choose two cut points
    i, j = rng.integers(0, n, 2)
    if i > j:
        i, j = j, i
    
    # Copy segment from p1
    child = np.full(n, -1, dtype=int)
    child[i:j+1] = p1[i:j+1]
    
    # Fill remaining positions with elements from p2 in order
    remaining = [x for x in p2 if x not in p1[i:j+1]]
    pos = 0
    for k in range(n):
        if child[k] == -1:
            child[k] = remaining[pos]
            pos += 1
    
    return child

def _ga_mutation(perm, rng, strength=0.1):
    """Apply random perturbation to a permutation."""
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    child = perm.copy()
    num_swaps = max(1, int(n * strength))
    for _ in range(num_swaps):
        i, j = rng.integers(0, n, 2)
        if i != j:
            child[i], child[j] = child[j], child[i]
    
    return child

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    effective_deadline = 145.0  # safety margin from 150s limit
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Initialize with NEH
    best_perm = _neh(processing_times)
    best_ms = _makespan_vec(best_perm, processing_times)
    
    # Create initial population with diverse solutions
    pop_size = min(20, max(5, n))
    population = [best_perm.copy()]
    
    # Add diverse initial solutions
    for _ in range(pop_size - 1):
        base = best_perm.copy()
        perturbed = _ga_mutation(base, rng, strength=0.3)
        polished = _insertion_ls(perturbed, processing_times, max_passes=2)
        population.append(polished)
    
    # Track best
    best_perm = population[0].copy()
    best_ms = _makespan_vec(best_perm, processing_times)
    for ind in population[1:]:
        ms = _makespan_vec(ind, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = ind.copy()
    
    # GA main loop
    gen = 0
    while time.time() < effective_deadline:
        gen += 1
        new_population = []
        
        # Elitism: keep best
        new_population.append(best_perm.copy())
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < effective_deadline:
            # Tournament selection
            candidates = []
            for _ in range(3):
                idx = rng.integers(0, len(population))
                candidates.append(population[idx])
            
            # Select best from tournament
            parent1 = min(candidates, key=lambda x: _makespan_vec(x, processing_times))
            
            # Second parent
            idx2 = rng.integers(0, len(population))
            parent2 = population[idx2]
            
            # Crossover
            child = _order_crossover(parent1, parent2, rng)
            
            # Mutation
            if rng.random() < 0.7:
                child = _ga_mutation(child, rng, strength=0.05)
            
            # Local search
            child = _insertion_ls(child, processing_times, max_passes=2)
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        for ind in population:
            ms = _makespan_vec(ind, processing_times)
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_perm = ind.copy()
        
        # Periodic restart to maintain diversity
        if gen % 10 == 0 and time.time() < effective_deadline:
            # Add some fresh NEH-based solutions
            for _ in range(min(3, pop_size // 4)):
                fresh = _neh(processing_times)
                fresh = _ga_mutation(fresh, rng, strength=0.2)
                fresh = _insertion_ls(fresh, processing_times, max_passes=2)
                if len(population) < pop_size:
                    population.append(fresh)
                else:
                    # Replace worst
                    worst_idx = max(range(len(population)), key=lambda i: _makespan_vec(population[i], processing_times))
                    population[worst_idx] = fresh
    
    return list(best_perm)
# EVOLVE-BLOCK-END
