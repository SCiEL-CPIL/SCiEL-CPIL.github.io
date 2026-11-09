"""Improved program for flow-shop permutation construction using diverse NEH variants and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Use list-based approach for small n_machines to reduce NumPy overhead
    if n_machines <= 10:
        # Pre-extract rows to avoid P[j, m] indexing overhead in inner loop?
        # P is a numpy array. P[j] is a view.
        # Let's just use lists for curr_row.
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
            
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                # max(curr_row[m], new_row[m-1]) + row[m]
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        # Use NumPy for larger M
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        
        return curr_row[-1]

def _neh_heuristic(P, sort_key='total', deadline=None, seed=None):
    """
    NEH heuristic for flow shop scheduling.
    sort_key: 'total' (sum), 'first_last' (P[:,0] + P[:,-1]), 'random'
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        # Add small random noise to break ties differently in randomized runs
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        # Heuristic: jobs with high processing on first and last machines are critical
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'random':
        if seed is not None:
            rng = np.random.RandomState(seed)
            sorted_indices = rng.permutation(n_jobs)
        else:
            sorted_indices = np.random.permutation(n_jobs)
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

def _local_search(order, P, deadline=None):
    """
    Local search using Or-opt moves (blocks of size 1, 2, 3) with First Improvement.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    # Use First Improvement strategy to maximize number of moves evaluated
    # Or-opt: remove a block of size k (1, 2, 3) and re-insert it
    block_sizes = [1, 2, 3]
    
    # We will loop until no improvement or deadline
    # To avoid infinite loops or getting stuck, we track if any improvement was made in a full pass
    while True:
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        
        # Iterate through all possible block starts and sizes
        # We need to be careful with indices when removing/inserting
        # It's safer to iterate i from 0 to n_jobs-1, and for each i, try block sizes
        # But if we modify current_order, indices change.
        # Standard approach:
        # For i in range(n_jobs):
        #   For k in block_sizes:
        #     If i+k <= n_jobs:
        #       block = current_order[i:i+k]
        #       rest = current_order[:i] + current_order[i+k:]
        #       For j in range(len(rest)+1):
        #          If j == i and (j+k-1 == i+k-1): continue # same position
        #          new_order = rest[:j] + block + rest[j:]
        #          new_ms = _makespan(new_order, P)
        #          if new_ms < current_ms:
        #             current_order = new_order
        #             current_ms = new_ms
        #             improved = True
        #             # Break out of j loop, and potentially i loop?
        #             # With First Improvement, we break j loop.
        #             # We should probably restart i loop or continue?
        #             # Continuing i loop with modified current_order is tricky.
        #             # Let's break out of all loops and restart the while loop.
        #             break
        #       else:
        #          continue
        #       break
        #
        #       # If no improvement for this i, continue to next i
        #
        # This is O(N^3 * K * Makespan) per pass.
        # With First Improvement, we stop as soon as we find an improvement.
        # This makes it very fast per improvement step.
        
        found_improvement = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
                
            for k in block_sizes:
                if i + k > n_jobs:
                    continue
                
                block = current_order[i:i+k]
                # Construct rest
                if i == 0:
                    rest = current_order[i+k:]
                elif i+k == n_jobs:
                    rest = current_order[:i]
                else:
                    rest = current_order[:i] + current_order[i+k:]
                
                # Try inserting block at all positions in rest
                # rest has length n_jobs - k
                # Positions 0 to n_jobs-k
                # We skip the position that results in the same sequence
                # The original position of the block was effectively i in the original list.
                # In the 'rest' list, the "gap" is at index i.
                # So we should skip j == i if we want to avoid no-op?
                # Actually, if we insert at j=i, we get the same sequence.
                # So we can skip j == i.
                
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                        
                    new_order = rest[:j] + block + rest[j:]
                    new_ms = _makespan(new_order, P)
                    
                    if new_ms < current_ms:
                        current_order = new_order
                        current_ms = new_ms
                        found_improvement = True
                        improved = True
                        # Break out of j, k, i loops
                        break
                else:
                    continue
                break
            else:
                continue
            break
            
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
        
    # 3. Reverse NEH (cheap diversification)
    if seq1:
        seq_rev = list(reversed(seq1))
        candidates.append(seq_rev)

    # 4. Randomized NEH (if n_jobs is small enough or time permits)
    if time.time() < deadline - 20.0 and n_jobs <= 200:
        # Use a fixed seed for reproducibility in this run, or random?
        # The task doesn't specify reproducibility, but deterministic is safer for debugging.
        # However, random restarts can help escape bad local minima.
        # Let's use a few random restarts if time is plentiful.
        for seed in [42, 123, 999]:
            if time.time() > deadline - 20.0:
                break
            seq_r = _neh_heuristic(P, sort_key='random', deadline=deadline, seed=seed)
            candidates.append(seq_r)

    # Find the best candidate
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    
    for seq in candidates[1:]:
        if len(seq) < n_jobs:
            # Complete if necessary
            remaining = [j for j in range(n_jobs) if j not in seq]
            seq.extend(remaining)
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq

    # Ensure best_seq is a valid permutation
    if len(best_seq) < n_jobs:
        remaining = [j for j in range(n_jobs) if j not in best_seq]
        best_seq.extend(remaining)
        
    # Apply local search if there is time
    if time.time() < deadline - 5.0:
        ls_deadline = deadline - 2.0
        best_seq = _local_search(best_seq, P, deadline=ls_deadline)
    
    return list(best_seq)

# EVOLVE-BLOCK-END
