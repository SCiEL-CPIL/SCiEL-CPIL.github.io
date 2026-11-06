"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """Calculate the makespan for a given job order in a flow shop."""
    if not order:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    curr = np.zeros(n_machines)
    for j in order:
        curr[0] += processing_times[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], curr[m]) + processing_times[j, m]
    return curr[-1]


def _makespan_from_state(order, processing_times, start_state):
    """Calculate makespan starting from a given completion state."""
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    curr = start_state.copy()
    for j in order:
        curr[0] += processing_times[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], curr[m]) + processing_times[j, m]
    return curr[-1]


def _compute_prefix_states(order, processing_times):
    """Precompute completion states after each position in order.
    Returns list where states[k] is the completion state after processing order[0:k].
    states[0] is zeros (before any job).
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    states = [None] * (n + 1)
    states[0] = np.zeros(n_machines)
    curr = np.zeros(n_machines)
    for k in range(n):
        j = order[k]
        curr[0] += processing_times[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], curr[m]) + processing_times[j, m]
        states[k+1] = curr.copy()
    return states


def _neh_heuristic(processing_times):
    """Generate initial permutation using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    job_sums = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-job_sums))
    
    current_sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_sequence.insert(best_pos, job)
    
    return current_sequence


def _fast_2opt(order, processing_times, deadline):
    """Fast 2-opt local search using prefix states."""
    n = len(order)
    n_machines = processing_times.shape[1]
    if n <= 1:
        return order, _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_makespan = current_makespan
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        states = _compute_prefix_states(current_order, processing_times)
        
        for i in range(n - 1):
            if time.time() > deadline:
                return best_order, best_makespan
            
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best_order, best_makespan
                
                # Reverse segment i..j
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                ms = _makespan(new_order, processing_times)
                
                if ms < current_makespan:
                    current_order = new_order
                    current_makespan = ms
                    if ms < best_makespan:
                        best_makespan = ms
                        best_order = current_order[:]
                    improved = True
    
    return best_order, best_makespan


def _fast_oropt(order, processing_times, deadline):
    """Or-opt local search: move a block of 1-3 jobs to another position."""
    n = len(order)
    n_machines = processing_times.shape[1]
    if n <= 2:
        return order, _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_makespan = current_makespan
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        states = _compute_prefix_states(current_order, processing_times)
        
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                return best_order, best_makespan
            
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    return best_order, best_makespan
                
                block = current_order[i:i+block_size]
                remaining = current_order[:i] + current_order[i+block_size:]
                
                best_pos = -1
                best_ms = current_makespan
                
                for j in range(len(remaining) + 1):
                    if time.time() > deadline:
                        return best_order, best_makespan
                    
                    if j == i or j == i + 1:
                        continue  # skip trivial positions
                    
                    new_order = remaining[:j] + block + remaining[j:]
                    ms = _makespan(new_order, processing_times)
                    
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                
                if best_pos != -1:
                    current_order = remaining[:best_pos] + block + remaining[best_pos:]
                    current_makespan = best_ms
                    if ms < best_makespan:
                        best_makespan = ms
                        best_order = current_order[:]
                    improved = True
                    break
            
            if improved:
                break
    
    return best_order, best_makespan


def _simulated_annealing(order, processing_times, start_temp, end_temp, deadline, rng):
    """Simulated annealing for flow shop."""
    n = len(order)
    n_machines = processing_times.shape[1]
    if n <= 1:
        return order, _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_makespan = current_makespan
    
    num_steps = max(1, int((deadline - time.time()) * 10000))  # estimate
    total_steps = max(100, num_steps)
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
        
        # Compute temperature
        progress = step / max(1, total_steps - 1)
        temp = start_temp * (end_temp / start_temp) ** progress
        if temp < 1e-10:
            temp = 1e-10
        
        # Generate neighbor: random swap or insertion
        if rng.random() < 0.5:
            # Swap two random positions
            i, j = rng.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Move job from i to j
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            new_order = current_order[:]
            job = new_order.pop(i)
            new_order.insert(j, job)
        
        new_makespan = _makespan(new_order, processing_times)
        delta = new_makespan - current_makespan
        
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_order = new_order
            current_makespan = new_makespan
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_order = current_order[:]
    
    return best_order, best_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Phase 1: NEH + fast local search
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Quick 2-opt pass
    local_deadline = start_time + 10.0
    if time.time() < local_deadline:
        current_order, current_makespan = _fast_2opt(current_order, processing_times, local_deadline)
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = current_order[:]
    
    # Quick Or-opt pass
    if time.time() < deadline - 5.0:
        current_order, current_makespan = _fast_oropt(current_order, processing_times, deadline - 5.0)
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = current_order[:]
    
    # Phase 2: Simulated Annealing with multiple restarts
    # Use remaining time for SA
    remaining_time = deadline - time.time()
    
    if remaining_time > 2.0:
        # Determine number of restarts based on time
        num_restarts = min(5, max(1, int(remaining_time / 5.0)))
        sa_deadline_per_restart = (deadline - time.time()) / max(1, num_restarts)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # For first restart, use best_order; for others, perturb
            if restart == 0:
                sa_start = best_order[:]
            else:
                # Perturb best_order with random swaps
                sa_start = best_order[:]
                num_swaps = max(1, n_jobs // 5)
                for _ in range(num_swaps):
                    i, j = rng.sample(range(n_jobs), 2)
                    sa_start[i], sa_start[j] = sa_start[j], sa_start[i]
            
            # Estimate makespan scale for temperature
            base_ms = _makespan(sa_start, processing_times)
            start_temp = base_ms * 0.1
            end_temp = base_ms * 0.001
            
            sa_order, sa_makespan = _simulated_annealing(
                sa_start, processing_times, start_temp, end_temp,
                time.time() + sa_deadline_per_restart, rng
            )
            
            if sa_makespan < best_makespan:
                best_makespan = sa_makespan
                best_order = sa_order[:]
    
    return best_order

# EVOLVE-BLOCK-END
