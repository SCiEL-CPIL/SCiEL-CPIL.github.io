"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # Cumulative completion times on each machine
    # C[j][m] = completion time of job j on machine m
    # For job j in position k, on machine m:
    #   C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # where C[j-1][m] is completion of previous job on same machine
    # We track only the previous job's completion per machine for efficiency
    
    # prev_machine_time[m] = completion time of last scheduled job on machine m
    # curr_job_time[m] = completion time of current job on machine m (built up)
    
    prev_machine_time = np.zeros(n_machines)
    makespan = 0.0
    for j in perm:
        curr_job_time = 0.0
        for m in range(n_machines):
            curr_job_time = max(curr_job_time, prev_machine_time[m]) + processing_times[j, m]
            prev_machine_time[m] = curr_job_time
        makespan = curr_job_time
    return makespan


def _insert_job(processing_times, perm, job_idx, pos):
    """Insert job_idx at position pos in perm, return new perm and makespan."""
    new_perm = list(perm[:pos]) + [job_idx] + list(perm[pos:])
    return new_perm, _makespan(processing_times, new_perm)


def _neighborhood_insertion(processing_times, perm):
    """Generate all single-job insertions (remove job i, insert at every position)."""
    n = len(perm)
    neighbors = []
    for i in range(n):
        job = perm[i]
        for pos in range(n + 1):
            if pos == i or pos == i + 1:
                continue
            new_perm = list(perm[:pos]) + [job] + list(perm[pos:])
            # Remove the original position
            # Wait, the above is wrong. Let's do it properly.
            # Remove job at i, then insert at pos (adjusting pos if pos > i)
            new_perm = list(perm[:i]) + list(perm[i+1:])
            insert_pos = pos if pos < i else pos - 1
            new_perm = new_perm[:insert_pos] + [job] + new_perm[insert_pos:]
            neighbors.append(new_perm)
    return neighbors


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    deadline = time.time() + 115.0  # Safety margin
    
    # Initial solution: NEH heuristic
    # Sort by total processing time in descending order
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    perm = list(order)
    best_perm = list(perm)
    best_makespan = _makespan(processing_times, perm)
    
    # Local search with insertion moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_makespan = best_makespan
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = best_perm[i]
            for pos in range(n_jobs + 1):
                if pos == i or pos == i + 1:
                    continue
                # Remove job at position i
                new_perm = list(best_perm[:i]) + list(best_perm[i+1:])
                insert_pos = pos if pos < i else pos - 1
                new_perm = new_perm[:insert_pos] + [job] + new_perm[insert_pos:]
                
                ms = _makespan(processing_times, new_perm)
                if ms < current_makespan:
                    current_makespan = ms
                    best_perm = new_perm
                    improved = True
                    break  # Restart inner loop with new best
            if improved:
                break
    
    return best_perm
# EVOLVE-BLOCK-END
