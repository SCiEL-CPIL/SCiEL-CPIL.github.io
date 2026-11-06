<<<<<<< SEARCH
def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times)
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # For small problems, use more thorough search
    if n_jobs <= 20:
        # Use multi-start with more iterations
        n_starts = 10
        time_per_start = min(10.0, (deadline - (time.time() - start_time)) / n_starts)
        result = _multi_start_local_search(processing_times, n_starts, time_per_start)
    else:
        # For larger problems, be more conservative
        n_starts = 5
        time_per_start = min(5.0, (deadline - (time.time() - start_time)) / n_starts)
        result = _multi_start_local_search(processing_times, n_starts, time_per_start)
    
    # Final check: ensure we're within time limit
    if time.time() - start_time > deadline - 2:
        # Return the best we have
        pass
    
    return result
# EVOLVE-BLOCK-END
=======
def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Determine number of starts based on problem size
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if n_jobs <= 15:
        n_starts = min(8, max(4, int(remaining / 15)))
        time_per_start = min(12.0, remaining / max(1, n_starts))
    elif n_jobs <= 30:
        n_starts = min(6, max(3, int(remaining / 20)))
        time_per_start = min(15.0, remaining / max(1, n_starts))
    else:
        n_starts = min(4, max(2, int(remaining / 30)))
        time_per_start = min(20.0, remaining / max(1, n_starts))
    
    result = _multi_start_local_search(
        processing_times, n_starts, time_per_start, start_time, deadline
    )
    
    # Final safety check
    if time.time() - start_time > deadline - 2:
        pass
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

