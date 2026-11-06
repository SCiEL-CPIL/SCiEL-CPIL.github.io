"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation efficiently."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    return machine_times[-1]


def _compute_completion_vector(perm, processing_times):
    """Compute the completion time vector for each machine."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return np.zeros(n_machines, dtype=np.float64)
    
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    return machine_times


def _heuristic_lower_bound(machine_times, remaining_jobs, processing_times):
    """
    A* heuristic: lower bound on additional time needed.
    For each remaining job, it must take at least its processing time on each machine.
    The last machine must process all remaining jobs sequentially.
    """
    n_machines = processing_times.shape[1]
    if not remaining_jobs:
        return 0.0
    
    # Lower bound: the last machine must process all remaining jobs
    # Plus the current time on the last machine
    remaining_on_last = sum(processing_times[j, n_machines - 1] for j in remaining_jobs)
    
    # Also, each machine must finish its remaining work
    # The makespan is at least max over machines of (current_time + remaining_work_on_that_machine)
    # But remaining work is distributed across machines based on which jobs are left
    
    # Simple admissible heuristic: max of (current machine time + min possible additional)
    # For the last machine: current + sum of processing times of remaining jobs on last machine
    lb_last = machine_times[-1] + remaining_on_last
    
    # For other machines, a simple bound is just their current time
    # A better bound: for each machine m, it needs to process all remaining jobs' m-th operation
    # but they can be interleaved. The minimum additional time for machine m is 
    # max(0, sum of pt[j][m] for remaining j - (slack from previous machines))
    # This is complex, so use a simpler bound:
    # The makespan >= max_m (machine_times[m] + min_pt_remaining_on_m)
    # where min_pt_remaining_on_m is the minimum processing time any remaining job has on machine m
    # But even simpler and still admissible:
    # makespan >= max_m (machine_times[m]) + min_j_in_remaining (sum_m pt[j][m])
    # This isn't quite right either.
    
    # Use a well-known admissible heuristic for flow shop:
    # h = max(
    #   max_m (machine_times[m] + sum_{j in remaining} pt[j][m] * (1/n_jobs_remaining)),  # not great
    #   machine_times[-1] + sum_{j in remaining} pt[j][-1]
    # )
    
    # Actually, the simplest admissible heuristic is:
    # The completion time on the last machine must be at least:
    #   max(machine_times[-1], max over machines of their current time) + 
    #   sum of processing times of all remaining jobs on the last machine
    # But this overestimates because jobs on earlier machines can overlap.
    
    # A correct admissible heuristic:
    # For the last machine, it cannot start processing job j before:
    #   max(machine_times[-1], completion of job j on machine m-1)
    # The earliest the last machine can finish all remaining jobs is bounded by:
    #   machine_times[-1] + sum_{j in remaining} pt[j][-1]
    # This is admissible because the last machine must process each remaining job's last operation.
    
    return lb_last


def _dp_beam_search(processing_times, beam_width=50, max_depth=None, time_limit=100.0):
    """
    Beam search over job subsets using DP-like state representation.
    State: (completed_mask, machine_times_tuple, path)
    We use beam search to keep only the best 'beam_width' states at each depth.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    
    # State: (makespan_estimate, machine_times, completed_mask, permutation)
    # We use a priority queue (min-heap) for A*-like search
    # But with beam pruning: at each depth, keep only top beam_width states
    
    # Initial state
    initial_mt = np.zeros(n_machines, dtype=np.float64)
    initial_state = (0.0, tuple(initial_mt), 0, ())
    
    # Use a list of (depth, priority, state) for beam management
    # At each depth, we expand all states in the beam and keep the best beam_width
    
    current_beam = [initial_state]
    best_complete = None
    best_complete_ms = float('inf')
    
    for depth in range(n_jobs):
        if time.time() - start_time > time_limit:
            break
        
        next_states = []
        
        for _, mt_tuple, mask, perm in current_beam:
            # Try adding each uncompleted job
            for job in range(n_jobs):
                if mask & (1 << job):
                    continue
                
                new_mask = mask | (1 << job)
                
                # Compute new machine times
                mt = np.array(mt_tuple, dtype=np.float64)
                pt = processing_times[job]
                prev = 0.0
                for m in range(n_machines):
                    new_time = max(mt[m], prev) + pt[m]
                    mt[m] = new_time
                    prev = new_time
                
                new_mt_tuple = tuple(mt)
                new_perm = perm + (job,)
                
                # Heuristic for A* ordering
                remaining_jobs = [j for j in range(n_jobs) if not (new_mask & (1 << j))]
                h = _heuristic_lower_bound(mt, remaining_jobs, processing_times)
                priority = mt[-1] + h  # f = g + h
                
                next_states.append((priority, new_mt_tuple, new_mask, new_perm))
        
        if not next_states:
            break
        
        # Sort by priority and keep top beam_width
        next_states.sort(key=lambda x: x[0])
        current_beam = next_states[:beam_width]
        
        # Check for complete solutions
        full_mask = (1 << n_jobs) - 1
        for priority, mt_tuple, mask, perm in current_beam:
            if mask == full_mask:
                ms = mt_tuple[-1]
                if ms < best_complete_ms:
                    best_complete_ms = ms
                    best_complete = list(perm)
        
        if time.time() - start_time > time_limit:
            break
    
    if best_complete is not None:
        return best_complete
    
    # If we didn't find a complete solution within time, return the best partial
    # extended greedily
    if current_beam:
        # Take the best state and greedily complete it
        _, mt_tuple, mask, perm = current_beam[0]
        mt = np.array(mt_tuple, dtype=np.float64)
        
        # Greedily add remaining jobs
        remaining = [j for j in range(n_jobs) if not (mask & (1 << j))]
        while remaining:
            best_job = None
            best_ms = float('inf')
            for job in remaining:
                mt_copy = mt.copy()
                pt = processing_times[job]
                prev = 0.0
                for m in range(n_machines):
                    new_time = max(mt_copy[m], prev) + pt[m]
                    mt_copy[m] = new_time
                    prev = new_time
                if mt_copy[-1] < best_ms:
                    best_ms = mt_copy[-1]
                    best_job = job
            perm = perm + (best_job,)
            mt = mt_copy
            remaining.remove(best_job)
        
        return list(perm)
    
    # Fallback: NEH
    return _neh_order(processing_times)


def _neh_order(processing_times, seed=None):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling with optional randomization."""
    n_jobs, n_machines = processing_times.shape
    rng = random.Random(seed)
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # For larger instances, add some randomization to the order
    if seed is not None and n_jobs > 10:
        # Slightly shuffle the sorted order
        for _ in range(min(5, n_jobs // 10)):
            idx = rng.randint(0, n_jobs - 2)
            sorted_jobs[idx], sorted_jobs[idx+1] = sorted_jobs[idx+1], sorted_jobs[idx]
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _local_search(perm, processing_times, time_limit=5.0):
    """
    Fast local search using only adjacent swaps.
    This is a simple but effective polish step.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            candidate = list(current_perm)
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_perm = candidate
                current_perm = candidate
                current_ms = ms
                improved = True
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses DP beam search (A*-inspired) for small/medium instances,
    with NEH fallback and local search polish.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (150s limit, use 135s)
    global_deadline = time.time() + 135.0
    
    best_perm = None
    best_ms = float('inf')
    
    # Strategy 1: DP beam search for small instances
    # Beam search is exact for n_jobs <= 12 with full beam, near-optimal for up to ~15
    if n_jobs <= 15:
        # Use a larger beam for smaller instances
        beam_width = min(200, 1 << n_jobs)  # Cap to avoid memory issues
        dp_time = min(60.0, (global_deadline - time.time()) * 0.6)
        
        dp_perm = _dp_beam_search(processing_times, beam_width=beam_width, time_limit=dp_time)
        if dp_perm:
            dp_ms = _makespan(dp_perm, processing_times)
            best_perm = dp_perm
            best_ms = dp_ms
    
    # Strategy 2: NEH as baseline or for larger instances
    neh_perm = _neh_order(processing_times, seed=None)
    neh_ms = _makespan(neh_perm, processing_times)
    
    if best_perm is None or neh_ms < best_ms:
        best_perm = neh_perm
        best_ms = neh_ms
    
    # Strategy 3: NEH with randomization for diversity
    if n_jobs > 5:
        neh_rand_perm = _neh_order(processing_times, seed=42)
        neh_rand_ms = _makespan(neh_rand_perm, processing_times)
        if neh_rand_ms < best_ms:
            best_perm = neh_rand_perm
            best_ms = neh_rand_ms
    
    # Strategy 4: Local search polish
    remaining = global_deadline - time.time()
    if remaining > 3.0:
        ls_time = min(remaining - 1.0, 40.0)
        improved_perm, improved_ms = _local_search(best_perm, processing_times, time_limit=ls_time)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # Strategy 5: If we have more time, try another NEH variant + local search
    remaining = global_deadline - time.time()
    if remaining > 5.0 and n_jobs > 5:
        # Try LPT (Longest Processing Time) order
        totals = processing_times.sum(axis=1)
        lpt_perm = list(np.argsort(-totals))
        lpt_ms = _makespan(lpt_perm, processing_times)
        
        if lpt_ms < best_ms:
            best_perm = lpt_perm
            best_ms = lpt_ms
        
        # Polish with local search
        remaining2 = global_deadline - time.time()
        if remaining2 > 3.0:
            ls_time2 = min(remaining2 - 1.0, 20.0)
            improved_perm2, improved_ms2 = _local_search(best_perm, processing_times, time_limit=ls_time2)
            if improved_ms2 < best_ms:
                best_perm = improved_perm2
                best_ms = improved_ms2
    
    return best_perm
# EVOLVE-BLOCK-END
