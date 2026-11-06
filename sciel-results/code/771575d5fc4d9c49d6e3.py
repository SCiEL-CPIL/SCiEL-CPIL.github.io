"""Improved program for permutation flow-shop construction using a search portfolio."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation on the given processing times matrix."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Use Python lists for faster access in tight loops
    # p_t is a list of lists for processing times
    p_t = processing_times.tolist()
    
    completion = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job_idx = perm[j]
        for m in range(n_machines):
            if m == 0:
                if j == 0:
                    completion[j, m] = p_t[job_idx][m]
                else:
                    completion[j, m] = completion[j-1, m] + p_t[job_idx][m]
            else:
                if j == 0:
                    completion[j, m] = p_t[job_idx][m]
                else:
                    completion[j, m] = max(completion[j-1, m] + p_t[job_idx][m],
                                           completion[j, m-1])
    return completion[-1, -1]


def _generate_neh_order(processing_times):
    """Generate NEH order using the full insertion heuristic."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    sorted_jobs = sorted(range(n_jobs), key=lambda j: -totals[j])
    
    current_perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_cost = float('inf')
        # Try inserting job at all positions
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            cost = _makespan(new_perm, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        current_perm.insert(best_pos, job)
        
    return current_perm


def _generate_random_order(processing_times):
    """Generate random order."""
    n_jobs = processing_times.shape[0]
    return list(np.random.permutation(n_jobs))


def _improve_order_insertion(order, processing_times, deadline):
    """Improve order using insertion moves (remove and re-insert)."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return list(order)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_order[i]
            # Remove job
            current_order.pop(i)
            best_pos = i
            best_cost = current_makespan
            
            # Try all insertion positions
            for pos in range(len(current_order) + 1):
                new_order = current_order[:pos] + [job] + current_order[pos:]
                new_makespan = _makespan(new_order, processing_times)
                if new_makespan < best_cost:
                    best_cost = new_makespan
                    best_pos = pos
            
            if best_cost < current_makespan:
                current_order.insert(best_pos, job)
                current_makespan = best_cost
                improved = True
            else:
                # Put it back where it was
                current_order.insert(i, job)
        if time.time() > deadline:
            break
    
    return current_order


def _improve_order_2opt(order, processing_times, deadline):
    """Improve order using 2-opt moves (reverse segment)."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return list(order)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() > deadline:
                    break
                # Try reversing segment [i, j]
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                new_makespan = _makespan(new_order, processing_times)
                if new_makespan < current_makespan:
                    current_order = new_order
                    current_makespan = new_makespan
                    improved = True
                    # Break inner loop to restart scan from beginning
                    break
            if improved:
                # If we made a change, break outer loop to restart scan
                break
    
    return current_order


def order_jobs(processing_times):
    """Main function to order jobs."""
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds deadline with 10 second safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    best_order = None
    best_makespan = float('inf')
    
    # Track 1: NEH + Insertion Local Search
    time_budget_1 = 30.0
    deadline_1 = start_time + time_budget_1
    neh_order = _generate_neh_order(processing_times)
    improved_order = _improve_order_insertion(neh_order, processing_times, deadline_1)
    ms = _makespan(improved_order, processing_times)
    if ms < best_makespan:
        best_makespan = ms
        best_order = improved_order
    
    if time.time() > deadline:
        return list(best_order)
    
    # Track 2: NEH + 2-Opt Local Search
    time_budget_2 = 30.0
    deadline_2 = start_time + time_budget_1 + time_budget_2
    improved_order = _improve_order_2opt(neh_order, processing_times, deadline_2)
    ms = _makespan(improved_order, processing_times)
    if ms < best_makespan:
        best_makespan = ms
        best_order = improved_order
    
    if time.time() > deadline:
        return list(best_order)
    
    # Track 3: Random + Insertion Local Search
    time_budget_3 = 30.0
    deadline_3 = start_time + time_budget_1 + time_budget_2 + time_budget_3
    random_order = _generate_random_order(processing_times)
    improved_order = _improve_order_insertion(random_order, processing_times, deadline_3)
    ms = _makespan(improved_order, processing_times)
    if ms < best_makespan:
        best_makespan = ms
        best_order = improved_order
    
    if time.time() > deadline:
        return list(best_order)
    
    # Track 4: Random + 2-Opt Local Search
    time_budget_4 = 30.0
    deadline_4 = start_time + time_budget_1 + time_budget_2 + time_budget_3 + time_budget_4
    improved_order = _improve_order_2opt(random_order, processing_times, deadline_4)
    ms = _makespan(improved_order, processing_times)
    if ms < best_makespan:
        best_makespan = ms
        best_order = improved_order
    
    if time.time() > deadline:
        return list(best_order)
    
    # Final polish: Run a quick insertion search on the best found solution with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        final_order = _improve_order_insertion(best_order, processing_times, deadline)
        ms = _makespan(final_order, processing_times)
        if ms < best_makespan:
            best_makespan = ms
            best_order = final_order
    
    return list(best_order)
# EVOLVE-BLOCK-END
