<<<<<<< SEARCH
def _local_search(processing_times, sequence, deadline):
    """
    First-improvement local search with insertion moves and limited block moves.
    Uses a time-based stopping criterion to balance exploration and exploitation.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    # Track how many consecutive non-improving evaluations we've done
    no_improve_count = 0
    max_no_improve = 2 * n_jobs * n_jobs  # Stop after this many non-improving tries
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try insertion moves (single job) - most effective neighborhood
        for i in range(n_jobs):
            if time.time() >= deadline or no_improve_count >= max_no_improve:
                return current_seq, current_makespan
            for j in range(n_jobs):
                if j == i or abs(i - j) == 1:
                    continue
                if time.time() >= deadline or no_improve_count >= max_no_improve:
                    return current_seq, current_makespan
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                if new_makespan < current_makespan:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    no_improve_count = 0
                    break  # restart scan from i=0
                else:
                    no_improve_count += 1
            if improved:
                break
        
        if improved:
            continue
        
        # Try block moves (2 jobs) if time permits and we haven't exhausted single-job moves
        if time.time() < deadline - 5.0 and n_jobs >= 4:
            for i in range(n_jobs - 1):
                if time.time() >= deadline or no_improve_count >= max_no_improve:
                    break
                block = current_seq[i:i+2]
                rest = current_seq[:i] + current_seq[i+2:]
                for j in range(len(rest) + 1):
                    if time.time() >= deadline or no_improve_count >= max_no_improve:
                        break
                    new_seq = rest[:j] + block + rest[j:]
                    if new_seq == current_seq:
                        continue
                    
                    new_makespan = _calc_makespan(processing_times, new_seq)
                    if new_makespan < current_makespan:
                        current_seq = new_seq
                        current_makespan = new_makespan
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        no_improve_count += 1
                if improved:
                    break
            if improved:
                continue

    return current_seq, current_makespan
=======
def _vns_search(processing_times, sequence, deadline, max_k=None):
    """
    Variable Neighborhood Search (VNS) that systematically explores increasing
    radii of the neighborhood by performing insertions with |i-j| <= k.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    if max_k is None:
        max_k = n_jobs - 1
    
    # Start with small neighborhoods and increase radius
    for k in range(1, min(max_k + 1, n_jobs)):
        if time.time() >= deadline:
            break
        
        found_improvement = False
        
        # Try all insertion moves with |i-j| <= k
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            
            # Check positions within radius k
            for delta in range(1, k + 1):
                # Try moving right
                j = i + delta
                if j < n_jobs:
                    new_seq = _insertion_move(current_seq, i, j)
                    new_makespan = _calc_makespan(processing_times, new_seq)
                    if new_makespan < current_makespan:
                        current_seq = new_seq
                        current_makespan = new_makespan
                        found_improvement = True
                        break  # Restart with this k
                # Try moving left
                j = i - delta
                if j >= 0:
                    new_seq = _insertion_move(current_seq, i, j)
                    new_makespan = _calc_makespan(processing_times, new_seq)
                    if new_makespan < current_makespan:
                        current_seq = new_seq
                        current_makespan = new_makespan
                        found_improvement = True
                        break  # Restart with this k
            if found_improvement:
                break
        
        # If no improvement at this radius, try next larger radius
        if not found_improvement:
            # Check if we've reached full neighborhood
            if k >= max_k:
                break
    
    return current_seq, current_makespan
>>>>>>> REPLACE

