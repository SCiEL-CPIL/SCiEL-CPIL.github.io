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

def _local_search(order, P, deadline=None, max_passes=10):
    """
    Local search using adjacent swaps (2-opt) and then insertions if time permits.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    # Phase 1: Adjacent Swaps
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try swapping adjacent jobs i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_ms = _makespan(new_order, P)
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
        
        if not improved:
            break
            
    # Phase 2: Insertion Moves (if time permits)
    # This is more expensive, so we limit it
    if deadline is not None and time.time() < deadline - 5.0:
        # Do a few passes of best-insertion
        for _pass in range(2): # Limit passes to keep it fast
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = current_order[i]
                # Remove job
                current_order.remove(job)
                best_pos = i
                best_ms = current_ms # Initial ms is for the sequence with job at i
                
                # Try inserting at all positions
                # Note: We removed job, so length is n_jobs-1. Positions 0 to n_jobs-1
                for pos in range(len(current_order) + 1):
                    if deadline is not None and pos % 5 == 0:
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
    deadline = start_time + 110.0
    
    # Generate multiple candidates using different NEH variants
    candidates = []
    
    # 1. Standard NEH
    seq1 = _neh_heuristic(P, sort_key='total', deadline=deadline)
    candidates.append(seq1)
    
    # 2. NEH with first/last machine focus
    if time.time() < deadline - 10.0: # Ensure we have time for another construction
        seq2 = _neh_heuristic(P, sort_key='first_last', deadline=deadline)
        candidates.append(seq2)
        
    # Select the best candidate based on makespan
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    
    for seq in candidates[1:]:
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
    
    # Apply local search to the best candidate
    if time.time() < deadline - 2.0: # Ensure we have time for local search
        improved_seq = _local_search(best_seq, P, deadline=deadline)
        improved_ms = _makespan(improved_seq, P)
        
        if improved_ms < best_ms:
            best_seq = improved_seq
            best_ms = improved_ms
    
    return list(best_seq)

# EVOLVE-BLOCK-END
