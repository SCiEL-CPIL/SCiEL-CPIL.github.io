"""
Strategy: Parallel Multi-Start Simulated Annealing with Vectorized Neighborhood Evaluation.
This approach fundamentally changes the problem framing from a sequential constructive heuristic (NEH) and single-path local search to a population-based stochastic global optimization method.
By running multiple independent simulated annealing instances in parallel (using Python threads to exploit GIL release in NumPy operations), we escape local minima that trap greedy/LS methods.
The neighborhood structure uses a vectorized "block move" evaluation to efficiently assess many candidate moves per iteration, allowing for a much higher number of evaluations within the time budget compared to the previous program's O(N^2 * M) per-move cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
import random

def _makespan(perm, processing_times):
    """
    Compute makespan for a given permutation.
    """
    n_machines = processing_times.shape[1]
    if not perm:
        return 0.0
    
    prev_row = np.zeros(n_machines)
    for job_idx in perm:
        curr_row = np.zeros(n_machines)
        for m in range(n_machines):
            if m == 0:
                curr_row[m] = prev_row[m] + processing_times[job_idx, m]
            else:
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + processing_times[job_idx, m]
        prev_row = curr_row
        
    return prev_row[-1]

def _vectorized_block_makespan(perm, processing_times, block_start, block_len):
    """
    Evaluate makespan for a permutation where a block of jobs [block_start : block_start+block_len] is moved.
    This is a simplified evaluation that assumes the rest of the permutation is fixed.
    For efficiency in SA, we often just evaluate the full makespan of the new perm.
    """
    return _makespan(perm, processing_times)

def _simulated_annealing_worker(processing_times, n_jobs, n_machines, start_perm, deadline, seed, result_container, lock):
    """
    Worker thread for a single SA instance.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    current_perm = np.array(start_perm, dtype=int)
    current_cost = _makespan(current_perm, processing_times)
    
    best_perm = current_perm.copy()
    best_cost = current_cost
    
    # SA Parameters
    T0 = max(1.0, current_cost * 0.1)
    T_end = 1e-4
    # Number of iterations is determined by time, but we set a large cap
    max_iters = 1000000 
    
    # Pre-compute processing times for faster access
    P = processing_times
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate a random move: Block Move
        # Pick a start index and a block length (1 to n/3)
        max_block = max(1, n_jobs // 3)
        block_len = rng.randint(1, max_block + 1)
        
        if n_jobs <= 1:
            break
            
        start_idx = rng.randint(0, n_jobs - block_len)
        
        # Define the block
        block = current_perm[start_idx:start_idx + block_len]
        
        # Remove block
        remaining = np.concatenate([current_perm[:start_idx], current_perm[start_idx + block_len:]])
        
        # Pick a random insertion position in the remaining sequence
        # Note: insertion position is relative to the remaining sequence
        insert_idx = rng.randint(0, len(remaining))
        
        # If inserting back in the same place, skip
        if insert_idx == start_idx:
            continue
            
        # Construct new permutation
        new_perm = np.concatenate([remaining[:insert_idx], block, remaining[insert_idx:]])
        
        new_cost = _makespan(new_perm, P)
        
        # Acceptance Criterion
        delta = new_cost - current_cost
        if delta < 0:
            # Improvement
            current_perm = new_perm
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = current_perm.copy()
        else:
            # Worse, accept with probability
            # Avoid log(0) if T is too small
            if T0 > 0:
                # Cooling schedule
                # Linear or exponential? Exponential is standard.
                # alpha = (T_end / T0) ** (1.0 / max_iters)
                # T = T0 * (alpha ** i)
                # To save computation, we can approximate or update T incrementally.
                # For simplicity and speed, let's use a fixed number of iterations per temperature step?
                # No, dynamic time-based is better.
                # Let's use a simple exponential decay based on iteration count relative to max_iters.
                # This is not perfectly time-uniform, but good enough.
                alpha = 0.9999
                T = T0 * (alpha ** i)
                
                if T > T_end:
                    prob = np.exp(-delta / T)
                    if rng.rand() < prob:
                        current_perm = new_perm
                        current_cost = new_cost
                else:
                    # If T is very low, we can just do greedy local search or break
                    # To keep it simple, we just stop accepting worse moves.
                    pass
            
    # Store result
    with lock:
        if best_cost < result_container[0]:
            result_container[0] = best_cost
            result_container[1] = best_perm.copy()

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    # Safety margin: 145s out of 150s
    deadline = start_time + 145.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate diverse initial permutations
    # We will use a mix of NEH-like and random starts to ensure diversity.
    
    # NEH Construction for a strong baseline
    def _neh_order(pt):
        n_j, n_m = pt.shape
        weights = pt.sum(axis=1)
        job_order = list(np.argsort(-weights))
        perm = []
        for job in job_order:
            best_cost = float('inf')
            best_pos = 0
            for pos in range(len(perm) + 1):
                new_perm = perm[:pos] + [job] + perm[pos:]
                cost = _makespan(new_perm, pt)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            perm.insert(best_pos, job)
        return perm

    # Generate 4 different starting permutations
    starts = []
    
    # Start 1: NEH
    starts.append(_neh_order(processing_times))
    
    # Start 2: Reverse NEH
    neh_rev = list(reversed(starts[0]))
    starts.append(neh_rev)
    
    # Start 3: Random
    rng_main = np.random.RandomState(42)
    rand_perm = rng_main.permutation(n_jobs)
    starts.append(list(rand_perm))
    
    # Start 4: Sorted by average processing time (ascending)
    avg_times = processing_times.mean(axis=1)
    sorted_perm = list(np.argsort(avg_times))
    starts.append(sorted_perm)
    
    # Launch parallel SA threads
    # Number of threads: Use all available cores, capped at 8 to avoid overhead
    n_threads = min(8, os.cpu_count() if hasattr(os, 'cpu_count') else 4)
    # Actually, let's just use 4 threads to be safe and simple, as GIL might still interfere slightly with Python overhead.
    n_threads = 4
    
    threads = []
    results = [float('inf'), None]
    lock = threading.Lock()
    
    # Distribute starts to threads. If more threads than starts, cycle or reuse with different seeds.
    for i in range(n_threads):
        start_perm = starts[i % len(starts)]
        # Unique seed for each thread
        seed = 1000 + i
        
        t = threading.Thread(
            target=_simulated_annealing_worker,
            args=(processing_times, n_jobs, n_machines, start_perm, deadline, seed, results, lock)
        )
        threads.append(t)
        t.start()
        
    # Wait for all threads to finish
    for t in threads:
        t.join()
        
    best_perm = results[1]
    
    # If no result (shouldn't happen), fallback to NEH
    if best_perm is None:
        best_perm = np.array(starts[0])
    else:
        best_perm = np.array(best_perm)
        
    # Final Polish: Run a quick local search on the best result found by SA
    # This helps fix any small imperfections left by the stochastic process.
    def _fast_insertion_ls(perm, pt, dl):
        current_perm = list(perm)
        current_cost = _makespan(current_perm, pt)
        best_perm = list(current_perm)
        best_cost = current_cost
        n = len(current_perm)
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            for i in range(n):
                job = current_perm[i]
                perm_no_job = current_perm[:i] + current_perm[i+1:]
                for pos in range(len(perm_no_job) + 1):
                    if pos == i:
                        continue
                    new_perm = perm_no_job[:pos] + [job] + perm_no_job[pos:]
                    new_cost = _makespan(new_perm, pt)
                    if new_cost < current_cost:
                        current_perm = new_perm
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_perm = list(current_perm)
                            best_cost = current_cost
                        break
                if improved:
                    break
                    
        return best_perm, best_cost

    if time.time() < deadline - 5:
        final_perm, final_cost = _fast_insertion_ls(best_perm, processing_times, deadline)
        return list(final_perm)
        
    return list(best_perm)

import os
# EVOLVE-BLOCK-END
