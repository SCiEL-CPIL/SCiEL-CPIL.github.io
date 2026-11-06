"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing.
    This approach uses stochastic acceptance of worse solutions to escape local minima,
    which is distinct from the deterministic local search in the previous version.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Convert to list of lists for faster access in Python loops
    pt_list = processing_times.tolist()
    
    # Initialize with a random permutation
    seq = list(range(n_jobs))
    random.shuffle(seq)
    
    # Helper to calculate makespan efficiently using lists
    def calc_makespan(perm):
        ct = [0.0] * n_machines
        for job_idx in perm:
            row = pt_list[job_idx]
            # Machine 0
            ct[0] += row[0]
            for m in range(1, n_machines):
                # ct[m] is prev job on machine m
                # ct[m-1] is current job on machine m-1
                if ct[m] < ct[m-1]:
                    ct[m] = ct[m-1] + row[m]
                else:
                    ct[m] += row[m]
        return ct[-1]

    current_makespan = calc_makespan(seq)
    best_seq = seq[:]
    best_makespan = current_makespan
    
    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 110.0  # Leave buffer for Python overhead
    # Heuristic initial temperature: scale with problem size and magnitude
    initial_temp = max(10.0, current_makespan * 0.5)
    cooling_rate = 0.995
    
    temp = initial_temp
    
    while temp > 1e-5:
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor: Insertion move
        i = random.randint(0, n_jobs - 1)
        job = seq[i]
        
        # Remove job
        seq_no_job = seq[:i] + seq[i+1:]
        # Insert at random position
        j = random.randint(0, len(seq_no_job))
        new_seq = seq_no_job[:j] + [job] + seq_no_job[j:]
        
        new_makespan = calc_makespan(new_seq)
        delta = new_makespan - current_makespan
        
        # Acceptance criterion
        if delta < 0:
            current_makespan = new_makespan
            seq = new_seq
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = seq[:]
        else:
            # Calculate probability
            # Avoid overflow in exp
            if delta / temp < 700: # exp(-700) is effectively 0
                prob = np.exp(-delta / temp)
                if random.random() < prob:
                    current_makespan = new_makespan
                    seq = new_seq
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_seq = seq[:]
        
        temp *= cooling_rate

    return best_seq
# EVOLVE-BLOCK-END
