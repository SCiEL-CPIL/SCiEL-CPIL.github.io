<<<<<<< SEARCH
def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    # perm is a list or array of job indices
    completion = [0.0] * n_machines
    
    for job in perm:
        # Access the row of processing times for this job
        row = processing_times[job]
        # Iterate over machines
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                # completion[m] = max(completion[m], completion[m-1]) + row[m]
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1]
=======
def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    # perm is a list or array of job indices
    completion = [0.0] * n_machines
    pt = processing_times
    
    for job in perm:
        row = pt[job]
        # Unroll first machine
        completion[0] += row[0]
        c_prev = completion[0]
        for m in range(1, n_machines):
            c_curr = completion[m]
            if c_curr < c_prev:
                c_curr = c_prev
            completion[m] = c_curr + row[m]
            c_prev = completion[m]
    
    return completion[-1]
>>>>>>> REPLACE

