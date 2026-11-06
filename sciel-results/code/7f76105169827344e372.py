"""Simulated Annealing for permutation flow-shop construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    n_machines = pt.shape[1]
    
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[perm[j], m]
    # We only need the completion time of the previous job on each machine.
    prev_row = np.zeros(n_machines)
    
    for j in range(n_jobs):
        job = perm[j]
        current_cum = 0.0
        for m in range(n_machines):
            current_cum = max(prev_row[m], current_cum) + pt[j, m]
            prev_row[m] = current_cum
            
    return prev_row[-1]

def _generate_neh_order(pt):
    """Generate an initial ordering using the NEH heuristic."""
    n_jobs, n_machines = pt.shape
    
    # Sum processing times for each job
    totals = pt.sum(axis=1)
    
    # Sort jobs in descending order of total processing time
    sorted_jobs = list(np.argsort(-totals))
    
    # Build the permutation incrementally using NEH
    perm = []
    
    for job in sorted_jobs:
        if not perm:
            perm.append(job)
            continue
        
        # Find the best position to insert this job
        best_pos = 0
        best_ms = float('inf')
        
        # Try all possible insertion positions
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm

def _simulated_annealing(perm, pt, time_limit):
    """Perform Simulated Annealing with adjacent swap moves."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    # Initial state
    current_perm = list(perm)
    current_ms = _makespan(current_perm, pt)
    
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Temperature parameters
    # T0 should be high enough to accept most moves initially.
    # Delta makespan is typically small relative to total makespan.
    # Let's estimate a typical delta based on max processing time.
    max_pt = np.max(pt)
    T0 = max(1.0, max_pt * 2.0) 
    T_final = 0.1
    
    start_time = time.time()
    
    # Cooling schedule: Geometric
    # We need to estimate the number of steps to determine alpha.
    # Let's run a quick benchmark to estimate steps per second.
    # Or just use a fixed number of steps if we can estimate time per step.
    # For n=100, m=10, step is ~100 ops. 1e7 steps ~ 1s?
    # Let's assume we can do ~1e6 to 1e7 steps in 100s.
    # Instead of fixed steps, we use time-based termination but need alpha.
    # Let's pick a reasonable number of iterations, say 100,000 to 1,000,000.
    # If n is small, we can do more. If n is large, fewer.
    
    # Estimate cost of one step: O(n*m) for makespan if we recompute, but we use delta?
    # Wait, I planned to use delta evaluation. Implementing exact delta for adjacent swap is complex.
    # Given the time constraints and the risk of delta bugs, let's stick to full makespan evaluation
    # but optimize the loop as much as possible.
    # Actually, for n=50, m=10, full makespan is 500 ops. 1e6 steps = 5e8 ops. ~0.5s in C, ~5s in Python.
    # So we can do roughly 1e5 - 1e6 steps in 100s depending on n.
    
    # Let's set a target number of steps based on n.
    target_steps = max(10000, int(5000000 / (n_jobs * pt.shape[1])))
    
    # Ensure we don't exceed time.
    # Alpha calculation: (T_final / T0)^(1/steps)
    alpha = (T_final / T0) ** (1.0 / target_steps)
    
    T = T0
    
    # Pre-allocate random number generator for speed
    rand = np.random.default_rng(42)
    
    for step in range(target_steps):
        # Time check every 1000 steps to avoid overhead
        if step % 1000 == 0:
            if time.time() - start_time > time_limit:
                break
        
        # Generate neighbor: swap adjacent jobs i and i+1
        i = rand.integers(0, n_jobs - 1)
        j = i + 1
        
        # Swap
        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        # Evaluate new makespan
        new_ms = _makespan(current_perm, pt)
        delta = new_ms - current_ms
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = list(current_perm)
        else:
            # Worse: accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
                if rand.random() < prob:
                    current_ms = new_ms
                else:
                    # Reject: swap back
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
        # Cool down
        T *= alpha
        
    return best_perm

def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard
    start_total = time.time()
    deadline = 140.0 # Safety margin from 150s kill
    
    # 1. Generate initial solution using NEH
    neh_perm = _generate_neh_order(processing_times)
    neh_ms = _makespan(neh_perm, processing_times)
    
    best_perm = neh_perm
    best_ms = neh_ms
    
    elapsed = time.time() - start_total
    time_left = deadline - elapsed
    
    if time_left > 1.0:
        # 2. Run Simulated Annealing
        sa_perm = _simulated_annealing(neh_perm, processing_times, time_left - 5.0)
        sa_ms = _makespan(sa_perm, processing_times)
        
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    return best_perm
# EVOLVE-BLOCK-END
