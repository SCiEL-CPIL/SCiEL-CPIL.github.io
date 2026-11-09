"""Improved program for flow-shop permutation construction using NEH heuristic with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Initialize for the first job in the sequence
    first_job = order[0]
    # For the first job, C(0, m) = sum(P[0, 0..m])
    curr_row = np.cumsum(P[first_job])
    
    for idx in range(1, n_jobs):
        j = order[idx]
        # curr_row was C(idx-1, m) for all m. Let's call it prev_row.
        prev_row = curr_row
        # Compute new curr_row for job j
        # C(j, 0) = C(j-1, 0) + P(j, 0)
        # C(j, m) = max(C(j-1, m), C(j, m-1)) + P(j, m)
        new_row = np.empty(n_machines)
        new_row[0] = prev_row[0] + P[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
        curr_row = new_row
    
    return curr_row[-1]

def _makespan_and_completion(order, P):
    """Compute makespan and full completion time matrix for a given job order."""
    if len(order) == 0:
        return 0.0, np.empty((0, P.shape[1]))
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0, np.empty((n_jobs, 0))
        
    C = np.empty((n_jobs, n_machines))
    first_job = order[0]
    C[0] = np.cumsum(P[first_job])
    
    for idx in range(1, n_jobs):
        j = order[idx]
        prev_row = C[idx - 1]
        C[idx, 0] = prev_row[0] + P[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                C[idx, m] = max(prev_row[m], C[idx, m-1]) + P[j, m]
    
    return C[-1, -1], C

def _neh_heuristic(P, sort_key='total', deadline=None):
    """
    NEH heuristic for flow shop scheduling.
    sort_key: 'total' (sum), 'first_last' (P[:,0] + P[:,-1])
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        # Heuristic: jobs with high processing on first and last machines are critical
        key_vals = P[:, 0] + P[:, -1]
        sorted_indices = np.argsort(-key_vals)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        # Check deadline
        if deadline is not None:
            if time.time() > deadline:
                break
        
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions (0 to len(current_seq))
        for pos in range(len(current_seq) + 1):
            # Check deadline inside inner loop for large n_jobs
            if deadline is not None and pos % 10 == 0:
                if time.time() > deadline:
                    break
            
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # If deadline was hit inside inner loop, best_pos might be -1 or partial
        if best_pos == -1:
            # If we couldn't find a valid position due to timeout, just append to end
            current_seq.append(job)
        else:
            # Insert at best position
            current_seq.insert(best_pos, job)
        
    return current_seq

def _local_search(order, P, deadline=None, max_passes=50):
    """
    Local search using adjacent swaps with delta evaluation and insertion moves.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms, current_C = _makespan_and_completion(current_order, P)
    
    # Phase 1: Adjacent Swaps with delta evaluation
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try swapping adjacent jobs i and i+1
            # Delta evaluation: only recompute rows from i to n_jobs-1
            j1 = current_order[i]
            j2 = current_order[i + 1]
            
            # Save original row i
            orig_row_i = current_C[i].copy()
            
            # Compute new row i (job j2 instead of j1)
            if i == 0:
                new_row_i = np.cumsum(P[j2])
            else:
                prev_row = current_C[i - 1]
                new_row_i = np.empty(P.shape[1])
                new_row_i[0] = prev_row[0] + P[j2, 0]
                for m in range(1, P.shape[1]):
                    new_row_i[m] = max(prev_row[m], new_row_i[m-1]) + P[j2, m]
            
            # Compute new row i+1 (job j1 instead of j2)
            new_row_i1 = np.empty(P.shape[1])
            new_row_i1[0] = new_row_i[0] + P[j1, 0]
            for m in range(1, P.shape[1]):
                new_row_i1[m] = max(new_row_i[m], new_row_i1[m-1]) + P[j1, m]
            
            # Check if makespan improves
            new_ms_candidate = new_row_i1[-1]
            
            if new_ms_candidate < current_ms:
                # Apply the swap
                current_order[i], current_order[i+1] = j2, j1
                current_C[i] = new_row_i
                current_C[i+1] = new_row_i1
                current_ms = new_ms_candidate
                
                # Update remaining rows if any
                for k in range(i + 2, n_jobs):
                    job_k = current_order[k]
                    prev_row = current_C[k-1]
                    current_C[k, 0] = prev_row[0] + P[job_k, 0]
                    for m in range(1, P.shape[1]):
                        current_C[k, m] = max(prev_row[m], current_C[k, m-1]) + P[job_k, m]
                
                improved = True
        
        if not improved:
            break
            
    # Phase 2: Insertion Moves (if time permits)
    if deadline is not None and time.time() < deadline - 3.0:
        for _pass in range(3):
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = current_order[i]
                # Remove job
                current_order.pop(i)
                best_pos = i
                best_ms = current_ms
                
                # Try inserting at all positions
                for pos in range(len(current_order) + 1):
                    if deadline is not None and pos % 10 == 0:
                        if time.time() > deadline:
                            break
                    new_order = current_order[:pos] + [job] + current_order[pos:]
                    new_ms = _makespan(new_order, P)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        best_pos = pos
                
                # Re-insert at best position
                current_order.insert(best_pos, job)
                if best_ms < current_ms:
                    current_ms = best_ms
                    improved = True
                    # Recompute C matrix
                    _, current_C = _makespan_and_completion(current_order, P)
            
            if not improved:
                break

    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses diverse NEH heuristics followed by local search improvement.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline with a safety margin
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple candidates using different NEH variants
    candidates = []
    
    # 1. Standard NEH
    seq1 = _neh_heuristic(P, sort_key='total', deadline=deadline)
    candidates.append(seq1)
    
    # 2. NEH with first/last machine focus
    if time.time() < deadline - 15.0:
        seq2 = _neh_heuristic(P, sort_key='first_last', deadline=deadline)
        candidates.append(seq2)
    
    # 3. NEH with first-machine priority
    if time.time() < deadline - 15.0:
        # Sort by processing time on first machine
        first_machine_times = P[:, 0]
        sorted_indices = np.argsort(-first_machine_times)
        current_seq = [int(sorted_indices[0])]
        for k in range(1, n_jobs):
            if time.time() > deadline:
                break
            job = int(sorted_indices[k])
            best_pos = -1
            best_makespan = float('inf')
            for pos in range(len(current_seq) + 1):
                if time.time() > deadline:
                    break
                new_seq = current_seq[:pos] + [job] + current_seq[pos:]
                ms = _makespan(new_seq, P)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            if best_pos == -1:
                current_seq.append(job)
            else:
                current_seq.insert(best_pos, job)
        candidates.append(current_seq)
        
    # Select the best candidate based on makespan
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    
    for seq in candidates[1:]:
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
    
    # Apply local search to the best candidate
    if time.time() < deadline - 1.0:
        improved_seq = _local_search(best_seq, P, deadline=deadline)
        improved_ms = _makespan(improved_seq, P)
        
        if improved_ms < best_ms:
            best_seq = improved_seq
            best_ms = improved_ms
    
    return list(best_seq)

# EVOLVE-BLOCK-END
