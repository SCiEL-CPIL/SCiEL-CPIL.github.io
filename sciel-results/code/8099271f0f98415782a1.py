<<<<<<< SEARCH
def _local_search(processing_times, sequence, deadline):
    """
    First-improvement insertion local search.
    Tries moving each job to every other position and accepts the first improvement.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                if new_makespan < current_makespan:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    break  # restart scan from i=0
            if improved:
                break
    
    return current_seq, current_makespan
=======
def _local_search(processing_times, sequence, deadline):
    """
    First-improvement local search with insertion and block moves (Or-opt).
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try insertion moves (single job)
        for i in range(n_jobs):
            if time.time() >= deadline:
                return current_seq, current_makespan
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    return current_seq, current_makespan
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                if new_makespan < current_makespan:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    break  # restart scan from i=0
            if improved:
                break
        
        if improved:
            continue
            
        # Try block moves (2 or 3 jobs) if time permits
        for block_size in [2, 3]:
            if time.time() >= deadline:
                return current_seq, current_makespan
            for i in range(n_jobs - block_size + 1):
                if time.time() >= deadline:
                    return current_seq, current_makespan
                block = current_seq[i:i+block_size]
                # Remove block
                rest = current_seq[:i] + current_seq[i+block_size:]
                # Insert block at various positions
                for j in range(len(rest) + 1):
                    if time.time() >= deadline:
                        return current_seq, current_makespan
                    new_seq = rest[:j] + block + rest[j:]
                    # Skip if it's the same position
                    if new_seq == current_seq:
                        continue
                    
                    new_makespan = _calc_makespan(processing_times, new_seq)
                    if new_makespan < current_makespan:
                        current_seq = new_seq
                        current_makespan = new_makespan
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return current_seq, current_makespan
>>>>>>> REPLACE

I should also update the SA to use the faster makespan and potentially allow it to run longer if the LS is faster.

