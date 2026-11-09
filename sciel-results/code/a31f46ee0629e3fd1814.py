"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m - 1] if m > 0 else 0) + processing_times[job, m]
    return completion[-1]


def _neh_init(processing_times):
    """NEH heuristic initialization."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job in the best position
    order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            new_order = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _greedy_insertion_init(processing_times):
    """Greedy insertion construction."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Start with the job with longest total processing time
    totals = processing_times.sum(axis=1)
    start_job = int(np.argmax(totals))
    order = [start_job]
    remaining = [j for j in range(n_jobs) if j != start_job]
    
    while remaining:
        best_job = None
        best_pos = 0
        best_ms = float('inf')
        for job in remaining:
            for pos in range(len(order) + 1):
                new_order = order[:pos] + [job] + order[pos:]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    best_job = job
                    best_pos = pos
        order = order[:best_pos] + [best_job] + order[best_pos:]
        remaining.remove(best_job)
    
    return order


def _local_search(processing_times, order, deadline):
    """Local search with adjacent swaps and reinsertion."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(processing_times, order)
        
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Adjacent swap moves
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            new_order = current_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            ms = _makespan(processing_times, new_order)
            if ms < current_ms - 1e-9:
                current_order = new_order
                current_ms = ms
                if ms < best_ms:
                    best_ms = ms
                    best_order = list(current_order)
                improved = True
        if time.time() > deadline:
            break
    
    # Reinsertion moves
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_order[i]
            base = current_order[:i] + current_order[i + 1:]
            for pos in range(len(base) + 1):
                if pos == i:
                    continue
                new_order = base[:pos] + [job] + base[pos:]
                ms = _makespan(processing_times, new_order)
                if ms < current_ms - 1e-9:
                    current_order = new_order
                    current_ms = ms
                    if ms < best_ms:
                        best_ms = ms
                        best_order = list(current_order)
                    improved = True
                    break
        if time.time() > deadline:
            break
    
    return best_order, best_ms


def _simulated_annealing(processing_times, order, deadline, rng):
    """Simulated Annealing with 2-opt block reversal neighborhood."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(processing_times, order)

    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms

    # Determine temperature schedule
    # Use a reasonable starting temperature based on problem size
    # Max possible makespan is roughly sum of all processing times
    max_ms = processing_times.sum()
    T = max_ms * 0.1
    T_min = max_ms * 1e-6
    alpha = 0.99  # Cooling rate
    
    # Estimate number of iterations possible
    # Rough estimate: each iteration involves one makespan calc
    # _makespan is O(n*m). If n=100, m=10, that's 1000 ops.
    # 1M iterations in 100s? Maybe too optimistic.
    # Let's just run until deadline.
    
    steps = 0
    while time.time() < deadline:
        # Generate a 2-opt move: reverse a block
        if n_jobs == 2:
            i = 0
            j = 1
        else:
            i = rng.integers(0, n_jobs - 1)
            j = rng.integers(i + 1, n_jobs)
        
        new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
        new_ms = _makespan(processing_times, new_order)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_order = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        steps += 1
        # Check time more frequently if steps are cheap
        if steps % 100 == 0 and time.time() > deadline:
            break
            
    return best_order, best_ms


def order_jobs(processing_times):
    """Main entry point: construct a job permutation minimizing makespan."""
    start_time = time.time()
    # Use 110s deadline to be safe within 120s limit (or 150s if that's the hard kill)
    # The prompt says "evaluator KILLS the program after 150s wall-clock"
    # But also "Each evaluation must finish within 120 s"
    # So 110s is a good safety margin.
    deadline = start_time + 110.0
    
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate multiple initial solutions
    candidates = []
    
    # NEH initialization
    neh_order = _neh_init(processing_times)
    candidates.append(neh_order)
    
    # Greedy insertion initialization
    gi_order = _greedy_insertion_init(processing_times)
    candidates.append(gi_order)
    
    # Simple LPT as fallback
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_order)
    
    # Run local search on each candidate within time budget
    best_order = None
    best_ms = float('inf')
    
    # Split time: 60% for local search on candidates, 40% for SA
    ls_deadline = start_time + (deadline - start_time) * 0.6
    
    for order in candidates:
        if time.time() > ls_deadline:
            break
        ls_order, ls_ms = _local_search(processing_times, order, ls_deadline)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    # If no local search was performed, use best candidate
    if best_order is None:
        for order in candidates:
            ms = _makespan(processing_times, order)
            if ms < best_ms:
                best_ms = ms
                best_order = order

    # Run Simulated Annealing from the best solution found
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        sa_order, sa_ms = _simulated_annealing(processing_times, best_order, deadline, rng)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_order = sa_order
            
        # If time remains, try SA from other candidates if they were significantly different
        # But usually the best LS solution is the best start for SA.
        # Let's try one more SA run from a different candidate if time permits and it was distinct
        if time.time() < deadline - 10:
            # Pick a candidate that is not the current best_order
            alt_candidate = None
            for c in candidates:
                if c != best_order:
                    alt_candidate = c
                    break
            if alt_candidate is not None:
                # Quick LS on the alternative to get a good starting point for SA
                if time.time() < deadline - 15:
                    ls_alt_order, ls_alt_ms = _local_search(processing_times, alt_candidate, deadline - 10)
                    if ls_alt_ms < best_ms:
                         best_ms = ls_alt_ms
                         best_order = ls_alt_order
                    # Run SA from this alternative
                    if time.time() < deadline - 2:
                        rng2 = np.random.default_rng(123)
                        sa2_order, sa2_ms = _simulated_annealing(processing_times, best_order, deadline, rng2)
                        if sa2_ms < best_ms:
                            best_ms = sa2_ms
                            best_order = sa2_order

    return best_order
# EVOLVE-BLOCK-END
