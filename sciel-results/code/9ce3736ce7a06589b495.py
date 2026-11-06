"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    Optimized for speed using local variables and minimal overhead.
    """
    if len(perm) == 0:
        return 0.0
    
    p = processing_times
    n_machines = p.shape[1]
    
    # C[m] is the completion time of the last job processed on machine m
    # We process jobs in the order of perm.
    # For a new job j:
    # prev_c = 0
    # for m in range(n_machines):
    #    new_c = max(C[m], prev_c) + p[j, m]
    #    C[m] = new_c
    #    prev_c = new_c
    
    C = np.zeros(n_machines, dtype=np.float64)
    
    # Iterate through jobs
    for j in perm:
        row = p[j]
        prev_c = 0.0
        for m in range(n_machines):
            # new_c = max(C[m], prev_c) + row[m]
            # Using if/else is faster than np.maximum for scalars in Python loops sometimes?
            # Actually, let's just use max() built-in or inline.
            if C[m] > prev_c:
                new_c = C[m] + row[m]
            else:
                new_c = prev_c + row[m]
            C[m] = new_c
            prev_c = new_c
            
    return C[-1]

def _calc_makespan_incremental(processing_times, perm, i, j, reverse=False):
    """
    Calculate makespan after swapping positions i and j in perm.
    perm is the list BEFORE the swap.
    This is an approximation/optimization: 
    The makespan is determined by the critical path.
    A swap only affects the schedule from min(i,j) to max(i,j) if we assume
    the completion times of machines before min(i,j) remain the same?
    NO. Swapping jobs changes the start times of subsequent jobs on all machines.
    So we must recalculate from min(i,j) onwards.
    
    However, for a 2-opt swap (swap i and j), the jobs between i and j are shifted.
    Let's implement a fast recalculation from the start of the affected region.
    
    Actually, a full recalculation is O(N*M).
    An incremental recalculation from position k is O((N-k)*M).
    If we swap i and j (i<j), the change starts at i.
    So we need the state of the completion times C at position i-1.
    
    Strategy:
    1. Compute C for the first i jobs.
    2. Simulate the rest of the permutation with the swap applied.
    
    This is still O(N*M) in the worst case but avoids the initial setup overhead 
    if we can cache C? No, C depends on the specific prefix.
    
    Let's stick to a robust full calculation but optimized, and rely on the 
    high iteration count of the population-based approach.
    """
    return _calc_makespan(processing_times, perm)

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm[:]
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # 2. Population-Based Iterated Local Search (ILS)
    # Maintain a population of K solutions
    K = 10  # Population size
    population = [best_perm[:]]
    pop_makespans = [best_ms]
    
    # Initialize other population members with slight perturbations of the best
    for _ in range(K - 1):
        perm = best_perm[:]
        # Random swap
        if n_jobs > 1:
            i, j = random.sample(range(n_jobs), 2)
            perm[i], perm[j] = perm[j], perm[i]
        ms = _calc_makespan(processing_times, perm)
        population.append(perm)
        pop_makespans.append(ms)
    
    # Local Search: Or-opt (move block of 1, 2, or 3 jobs)
    def local_search(perm, ms, time_limit=0.02):
        """
        Perform Or-opt local search on perm until no improvement or time limit.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(perm)
            # Try moving blocks of size 1, 2, 3
            for block_size in [1, 2, 3]:
                if n < block_size:
                    continue
                for i in range(n - block_size + 1):
                    block = perm[i:i+block_size]
                    # Remove block
                    rem_perm = perm[:i] + perm[i+block_size:]
                    # Try inserting at all positions
                    for j in range(len(rem_perm) + 1):
                        # Avoid inserting back to same place (roughly)
                        if j == i or (j == i + block_size):
                            continue
                        new_perm = rem_perm[:j] + block + rem_perm[j:]
                        new_ms = _calc_makespan(processing_times, new_perm)
                        if new_ms < ms:
                            perm = new_perm
                            ms = new_ms
                            improved = True
                            # Restart local search? Or continue? 
                            # Usually restart or continue with new base.
                            # Let's restart the while loop to re-evaluate all moves from new state
                            break
                    if improved:
                        break
                if improved:
                    break
        return perm, ms
    
    # Main ILS Loop
    while time.time() < deadline:
        # For each individual in population
        for idx in range(K):
            if time.time() > deadline - 2.0:
                break
            
            curr_perm = population[idx][:]
            curr_ms = pop_makespans[idx]
            
            # 1. Local Search
            curr_perm, curr_ms = local_search(curr_perm, curr_ms, time_limit=deadline - 5.0)
            
            # Update population
            if curr_ms < pop_makespans[idx]:
                population[idx] = curr_perm
                pop_makespans[idx] = curr_ms
                
                # Update global best
                if curr_ms < best_ms:
                    best_ms = curr_ms
                    best_perm = curr_perm[:]
            
            # 2. Perturbation (if not best)
            if population[idx] != best_perm or pop_makespans[idx] > best_ms:
                # Perturb the current solution
                # Or-opt style perturbation: move a random block
                if n_jobs > 3:
                    block_size = random.choice([1, 2, 3])
                    i = random.randint(0, n_jobs - block_size)
                    block = curr_perm[i:i+block_size]
                    rem_perm = curr_perm[:i] + curr_perm[i+block_size:]
                    j = random.randint(0, len(rem_perm))
                    new_perm = rem_perm[:j] + block + rem_perm[j:]
                else:
                    # Small instance: random swap
                    i, j = random.sample(range(n_jobs), 2)
                    new_perm = curr_perm[:]
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                
                new_ms = _calc_makespan(processing_times, new_perm)
                
                # Acceptance criterion: 
                # If new is better than current, accept.
                # If new is better than best, accept.
                # Otherwise, accept with some probability to maintain diversity?
                # Simple: Replace if better than current, or if random < 0.1 (restart diversity)
                if new_ms < curr_ms or random.random() < 0.05:
                    population[idx] = new_perm
                    pop_makespans[idx] = new_ms
                else:
                    # Keep the locally optimized version
                    pass
            
            # 3. Selection/Replacement
            # If this individual is worse than the worst in the population, 
            # and the best individual is much better, consider replacing?
            # For simplicity, we just let the population evolve.
            # However, to prevent premature convergence, we can occasionally 
            # replace the worst individual with a perturbed version of the best.
            
            if idx == K - 1: # Do this once per cycle
                worst_idx = np.argmax(pop_makespans)
                if pop_makespans[worst_idx] > best_ms * 1.05: # If worst is >5% worse than best
                    # Replace worst with perturbed best
                    perturbed = best_perm[:]
                    if n_jobs > 1:
                        # Strong perturbation: Or-opt move
                        block_size = random.choice([1, 2, 3, 4, 5])
                        if n_jobs >= block_size:
                            i = random.randint(0, n_jobs - block_size)
                            block = perturbed[i:i+block_size]
                            rem = perturbed[:i] + perturbed[i+block_size:]
                            j = random.randint(0, len(rem))
                            perturbed = rem[:j] + block + rem[j:]
                        else:
                            i, j = random.sample(range(n_jobs), 2)
                            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    
                    p_ms = _calc_makespan(processing_times, perturbed)
                    population[worst_idx] = perturbed
                    pop_makespans[worst_idx] = p_ms

    return best_perm
# EVOLVE-BLOCK-END
