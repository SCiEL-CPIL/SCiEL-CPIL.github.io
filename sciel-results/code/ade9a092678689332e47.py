"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion[j][m] = completion time of job j on machine m
    # We compute iteratively
    # completion time of job i on machine m:
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + P[i][m]
    # where C[-1][m] = 0 and C[i][-1] = 0
    
    # Use a 2D array or just track the last machine completion for each job
    # Actually, we need C[i][m-1] which is the completion of the same job on previous machine
    # and C[i-1][m] which is the completion of previous job on same machine.
    
    # Let's use a 1D array for the current job's completions across machines,
    # and a 1D array for the previous job's completions (which is the machine completion times).
    
    # machine_completion[m] = completion time of the last scheduled job on machine m
    machine_completion = np.zeros(n_machines)
    
    for job_idx in order:
        prev_job_completion = 0.0  # C[i][m-1], starts at 0 for first machine
        for m in range(n_machines):
            # C[i][m] = max(machine_completion[m], prev_job_completion) + P[i][m]
            current = max(machine_completion[m], prev_job_completion) + processing_times[job_idx, m]
            machine_completion[m] = current
            prev_job_completion = current
    
    return machine_completion[-1]


def _neh_order(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    # Sort jobs by decreasing total processing time
    sorted_jobs = list(np.argsort(-totals))
    
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(sequence) + 1):
            trial = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _two_opt(processing_times, sequence, max_iters=1000, time_limit=5.0):
    """2-opt local search on the sequence."""
    n = len(sequence)
    if n <= 2:
        return sequence
    
    current_seq = list(sequence)
    current_ms = _makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    start_time = time.time()
    no_improve = 0
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse the segment from i to j
                trial = current_seq[:i] + current_seq[i:j+1][::-1] + current_seq[j+1:]
                trial_ms = _makespan(processing_times, trial)
                if trial_ms < current_ms - 1e-9:
                    current_seq = trial
                    current_ms = trial_ms
                    if current_ms < best_ms - 1e-9:
                        best_seq = list(current_seq)
                        best_ms = current_ms
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 5:
                break
    
    return best_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + 2-opt local search.
    """
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin
    
    # Get initial solution using NEH
    initial_seq = _neh_order(processing_times)
    best_seq = list(initial_seq)
    best_ms = _makespan(processing_times, best_seq)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_seq
    
    # Apply 2-opt local search
    remaining_time = deadline - elapsed
    if remaining_time > 0.1:
        improved_seq = _two_opt(processing_times, best_seq, max_iters=5000, time_limit=remaining_time * 0.8)
        improved_ms = _makespan(processing_times, improved_seq)
        if improved_ms < best_ms:
            best_seq = improved_seq
            best_ms = improved_ms
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_seq
    
    # If we have more time and the problem is small enough, try a few more restarts
    # with perturbed initial solutions
    remaining_time = deadline - elapsed
    if remaining_time > 2.0 and n_jobs <= 30:
        # Try a few perturbed NEH solutions
        for _ in range(3):
            if time.time() - start_time > deadline - 1.0:
                break
            
            # Perturb: swap two random jobs in the NEH sequence
            perturbed = list(initial_seq)
            i, j = np.random.choice(n_jobs, 2, replace=False)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            local_seq = _two_opt(processing_times, perturbed, max_iters=2000, 
                                  time_limit=min(remaining_time * 0.2, 5.0))
            local_ms = _makespan(processing_times, local_seq)
            if local_ms < best_ms:
                best_seq = local_seq
                best_ms = local_ms
                remaining_time = deadline - (time.time() - start_time)
    
    return best_seq
# EVOLVE-BLOCK-END
