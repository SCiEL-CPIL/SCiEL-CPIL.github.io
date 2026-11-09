"""Flow-shop permutation construction using NEH + tabu search + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _makespan_from_positions(positions, P):
    """Compute makespan given positions array where positions[j] = position of job j."""
    order = np.argsort(positions)
    return _makespan(order, P)


def _neh_heuristic(P):
    """NEH heuristic: build permutation by inserting each job at best position."""
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    totals = P.sum(axis=1)
    # Sort jobs by descending total processing time
    sorted_jobs = np.argsort(-totals)
    
    # Build permutation incrementally
    perm = []
    for j in sorted_jobs:
        # Try inserting job j at each possible position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, j)
    
    return perm


def _two_opt(order, P, max_iter=1000):
    """2-opt local search: try swapping pairs of jobs."""
    n = len(order)
    if n <= 2:
        return order, _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing segment i..j
                candidate = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                ms = _makespan(candidate, P)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return best_order, best_ms


def _or_opt(order, P, max_iter=500):
    """Or-opt local search: try moving 1 or 2 consecutive jobs to another position."""
    n = len(order)
    if n <= 3:
        return order, _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for length in [1, 2]:
            for i in range(n - length + 1):
                segment = best_order[i:i+length]
                # Remove segment
                remaining = best_order[:i] + best_order[i+length:]
                # Try inserting at each position in remaining
                for j in range(len(remaining) - length + 1):
                    if j == i:
                        continue
                    candidate = remaining[:j] + segment + remaining[j:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-10:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_order, best_ms


def _weights_to_order(weights, P):
    """Convert weight vector to job order by sorting weights in descending order."""
    n = len(weights)
    # Sort jobs by descending weight
    order = list(np.argsort(-weights))
    return order


def _tabu_search(order, P, max_iter=500, tabu_len=15, time_limit=30.0):
    """Tabu search using priority-list representation.
    
    Uses a continuous weight vector where the order is determined by sorting weights.
    Moves perturb the weights to explore non-local optima.
    """
    n = len(order)
    if n <= 5:
        return order, _makespan(order, P)
    
    # Initialize weights from current order
    weights = np.zeros(n, dtype=np.float64)
    for pos, job in enumerate(order):
        weights[job] = n - pos  # Higher weight = earlier position
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    current_order = list(order)
    current_ms = best_ms
    
    tabu_list = {}  # job -> iteration when it was last moved
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iter:
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        iteration += 1
        
        # Generate candidate moves: perturb weights
        best_candidate_order = None
        best_candidate_ms = float('inf')
        best_move = None
        
        # Try several random perturbations
        for _ in range(20):
            new_weights = weights.copy()
            
            # Pick 1-3 jobs to perturb
            n_perturb = random.randint(1, min(3, n))
            jobs_to_perturb = random.sample(range(n), n_perturb)
            
            for job in jobs_to_perturb:
                # Check tabu
                if job in tabu_list and iteration - tabu_list[job] < tabu_len:
                    continue
                # Perturb weight
                new_weights[job] += random.uniform(-2.0, 2.0)
            
            # Convert to order
            candidate_order = _weights_to_order(new_weights, P)
            
            # Skip if same as current
            if candidate_order == current_order:
                continue
            
            ms = _makespan(candidate_order, P)
            
            # Accept if better or if not tabu (aspiration)
            if ms < best_candidate_ms:
                best_candidate_ms = ms
                best_candidate_order = candidate_order
                best_move = (jobs_to_perturb, new_weights)
        
        if best_candidate_order is None:
            # No valid move found, break
            break
        
        # Check if this is an improving move or if we should accept anyway (diversification)
        is_improving = best_candidate_ms < current_ms - 1e-10
        
        # Accept if improving, or if it's the best tabu-breaking move
        if is_improving or best_candidate_ms < best_ms:
            # Update current
            current_order = best_candidate_order
            current_ms = best_candidate_ms
            
            # Update tabu list
            if best_move:
                for job in best_move[0]:
                    tabu_list[job] = iteration
                weights = best_move[1]
            
            # Update best
            if current_ms < best_ms - 1e-10:
                best_ms = current_ms
                best_order = list(current_order)
        else:
            # No improvement, try a more aggressive perturbation
            # Pick 2-4 random jobs and swap their positions
            n_swap = random.randint(2, min(4, n))
            jobs_to_swap = random.sample(range(n), n_swap)
            
            # Get positions of these jobs in current order
            pos_in_order = {job: i for i, job in enumerate(current_order)}
            
            # Swap their positions randomly
            new_positions = list(pos_in_order[j] for j in jobs_to_swap)
            random.shuffle(new_positions)
            
            # Build new order
            candidate = list(current_order)
            for i, job in enumerate(jobs_to_swap):
                candidate[new_positions[i]] = job
            
            ms = _makespan(candidate, P)
            
            if ms < best_ms - 1e-10:
                best_ms = ms
                best_order = list(candidate)
            
            current_order = candidate
            current_ms = ms
            
            # Update tabu
            for job in jobs_to_swap:
                tabu_list[job] = iteration
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 110s out of 150s limit
    
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    # Start with NEH heuristic
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    # Apply 2-opt local search
    best_order, best_ms = _two_opt(best_order, P, max_iter=200)
    
    # Check time
    if time.time() > deadline:
        return best_order
    
    # Apply Or-opt local search
    best_order, best_ms = _or_opt(best_order, P, max_iter=100)
    
    # Check time
    if time.time() > deadline:
        return best_order
    
    # Apply tabu search for diversification
    time_remaining = deadline - time.time()
    if time_remaining > 30:
        tabu_time = min(30.0, time_remaining - 10)
        best_order, best_ms = _tabu_search(best_order, P, max_iter=300, tabu_len=15, time_limit=tabu_time)
    
    # Check time
    if time.time() > deadline:
        return best_order
    
    # Final local search polish
    if time.time() < deadline - 15:
        best_order, best_ms = _two_opt(best_order, P, max_iter=100)
    
    if time.time() < deadline - 5:
        best_order, best_ms = _or_opt(best_order, P, max_iter=50)
    
    return best_order
# EVOLVE-BLOCK-END
