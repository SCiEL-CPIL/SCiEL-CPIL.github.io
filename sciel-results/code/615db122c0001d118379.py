"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _compute_c_matrix(processing_times, order):
    """Compute the full C[i][j] matrix for the given order."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines))
    if n_jobs == 0:
        return C
    for i, job_idx in enumerate(order):
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = C[i-1, j] if i > 0 else 0.0
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            C[i, j] = c_val
            prev_c = c_val
    return C

def _insertion_delta(processing_times, order, C, job_idx, from_pos, to_pos):
    """
    Compute the new makespan after moving job at from_pos to to_pos.
    Uses O(M) complexity for the affected segment.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if from_pos == to_pos:
        return C[n_jobs - 1, n_machines - 1]
    
    # Determine the range of affected jobs
    # If moving forward: from_pos to to_pos-1 are shifted right
    # If moving backward: from_pos+1 to to_pos are shifted left
    # We need to recompute completion times for the segment between from_pos and to_pos inclusive (in the new order)
    
    # To keep it simple and robust, we identify the start and end of the affected block in the NEW order.
    # The jobs that change their relative position with respect to the moved job are those between from_pos and to_pos.
    
    # Let's reconstruct the relevant part of the C matrix.
    # The unaffected prefix is up to min(from_pos, to_pos) - 1.
    # The unaffected suffix is from max(from_pos, to_pos) + 1 onwards (but their start times depend on the prefix).
    
    # Actually, the standard efficient delta calculation for insertion:
    # 1. Remove job at from_pos.
    # 2. Insert at to_pos.
    # The completion times of jobs before min(from, to) are unchanged.
    # The completion times of jobs after max(from, to) are shifted by the change in the completion time of the last job in the affected block.
    # Wait, that's not quite right. The completion time of a job depends on the previous job on the same machine AND the previous machine.
    
    # Let's use a safer O(N*M) re-evaluation for the affected segment, which is much faster than O(N^2*M) if the segment is small, 
    # but in worst case (moving end to start) it's O(N*M). 
    # Given N is likely small (flow shop instances are often <= 100), O(N*M) per move is acceptable if we limit the number of moves.
    # However, the previous code tried to be clever and failed. Let's do a straightforward re-calculation of the suffix.
    
    # Identify the start index in the new order where changes begin.
    # If from < to: The job moves right. The jobs from_pos..to_pos-1 shift left. 
    #   New order indices: 
    #   0..from-1: same
    #   from..to-1: old from+1..to
    #   to: old from
    #   to+1..end: same
    #   The completion times from index `from` onwards need to be recalculated.
    # If from > to: The job moves left. The jobs to..from-1 shift right.
    #   New order indices:
    #   0..to-1: same
    #   to: old from
    #   to+1..from: old to..from-1
    #   from+1..end: same
    #   The completion times from index `to` onwards need to be recalculated.
    
    start_recalc = min(from_pos, to_pos)
    
    # Create a temporary new order for the calculation
    new_order = list(order)
    job = new_order.pop(from_pos)
    new_order.insert(to_pos, job)
    
    # We need the completion time of the job before start_recalc in the NEW order to start the chain.
    # If start_recalc == 0, prev_C is 0.
    # Else, prev_C is C_new[start_recalc - 1]. Since indices < start_recalc are unchanged in value (though indices might shift if from<to? No, indices < min(from,to) are identical in value and position).
    
    if start_recalc == 0:
        prev_row = np.zeros(n_machines)
    else:
        prev_row = C[start_recalc - 1] # This is valid because the prefix is unchanged
    
    new_C_suffix = np.zeros((n_jobs - start_recalc, n_machines))
    
    for i in range(start_recalc, n_jobs):
        idx_in_new = i - start_recalc
        j_idx = new_order[i]
        p_row = processing_times[j_idx]
        
        # Get previous machine completion from the new suffix calculation or the original prefix
        if i == start_recalc:
            prev_machine_times = prev_row
        else:
            prev_machine_times = new_C_suffix[idx_in_new - 1]
            
        current_row = np.zeros(n_machines)
        prev_c = 0.0
        for j in range(n_machines):
            c_val = prev_machine_times[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            current_row[j] = c_val
            prev_c = c_val
            
        new_C_suffix[idx_in_new] = current_row
    
    return new_C_suffix[-1, -1]

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    # Set a safe deadline well before 150s
    deadline = start_time + 110
    
    # --- Phase 1: NEH Constructive Heuristic ---
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    current_order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        # Optimize NEH insertion: only check positions in the current partial sequence
        for pos in range(len(current_order) + 1):
            # Quick check: if pos is 0 or len, we can estimate, but full calc is safer
            # For small n, this is fine.
            new_order = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order.insert(best_pos, job)
        
        if time.time() > deadline:
            break
    
    best_order = list(current_order)
    best_ms = _makespan(processing_times, best_order)
    C_best = _compute_c_matrix(processing_times, best_order)
    
    # --- Phase 2: Iterated Local Search (ILS) ---
    # ILS is robust and fast. We use insertion moves.
    current_order = list(best_order)
    current_ms = best_ms
    C_current = _compute_c_matrix(processing_times, current_order)
    
    n = len(current_order)
    
    while time.time() < deadline:
        # Local Search: Best Improvement Insertion
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_move_ms = current_ms
            best_move_from = -1
            best_move_to = -1
            
            # Sample a subset of moves if n is large to save time, otherwise all
            # For n < 50, check all. For larger, check a random subset.
            if n < 50:
                moves_to_check = [(i, j) for i in range(n) for j in range(n+1) if i != j and i != j-1]
            else:
                # Random sampling of moves
                moves_to_check = [(random.randint(0, n-1), random.randint(0, n)) for _ in range(n*5)]
                # Filter invalid
                moves_to_check = [(i,j) for i,j in moves_to_check if i != j and i != j-1]
                
            for i in moves_to_check:
                from_pos, to_pos = i
                if to_pos == from_pos or to_pos == from_pos + 1:
                    continue
                    
                new_ms = _insertion_delta(processing_times, current_order, C_current, current_order[from_pos], from_pos, to_pos)
                
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move_from = from_pos
                    best_move_to = to_pos
                    improved = True
            
            if improved:
                # Apply best move
                job = current_order.pop(best_move_from)
                current_order.insert(best_move_to, job)
                current_ms = best_move_ms
                C_current = _compute_c_matrix(processing_times, current_order)
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current_order)
                    C_best = _compute_c_matrix(processing_times, best_order)
        
        # Perturbation
        if time.time() > deadline:
            break
            
        # Random Or-opt perturbation: move a small segment (1-3 jobs) to a random position
        seg_len = random.randint(1, min(3, n))
        start_idx = random.randint(0, n - seg_len)
        segment = current_order[start_idx:start_idx+seg_len]
        
        # Remove segment
        del current_order[start_idx:start_idx+seg_len]
        
        # Insert at random position
        insert_pos = random.randint(0, len(current_order))
        current_order[insert_pos:insert_pos] = segment
        
        current_ms = _makespan(processing_times, current_order)
        C_current = _compute_c_matrix(processing_times, current_order)
        
        # Accept if better or with some probability (SA-like) or just always accept to restart LS
        # Standard ILS accepts if better, or with probability p. 
        # Simple ILS: accept if better. If not, we still restart LS from this perturbed state? 
        # Actually, standard ILS: 
        # 1. LS
        # 2. Perturb
        # 3. LS
        # We always proceed to LS regardless of acceptance? 
        # Usually, we accept the perturbed solution if it's not too bad, or use SA criteria.
        # Let's use a simple rule: If current_ms < best_ms * 1.1, accept. Else, revert? 
        # To keep it simple and ensure exploration, we just restart LS from the perturbed state.
        
    return best_order
# EVOLVE-BLOCK-END
