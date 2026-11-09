# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_makespan_matrix(perm, processing_times):
    """
    Compute the makespan and the completion time matrix for a given permutation.
    Returns (makespan, completion_matrix).
    completion_matrix[j][m] is the completion time of job perm[j] on machine m.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0, np.zeros((0, n_machines))
        
    # C[j][m] = completion time of job perm[j] on machine m
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[perm[j], m]
    
    C = np.zeros((n_jobs, n_machines))
    
    for j_idx in range(n_jobs):
        job = perm[j_idx]
        if j_idx == 0:
            # First job
            C[j_idx, 0] = processing_times[job, 0]
            for m in range(1, n_machines):
                C[j_idx, m] = C[j_idx, m-1] + processing_times[job, m]
        else:
            # Subsequent jobs
            C[j_idx, 0] = C[j_idx-1, 0] + processing_times[job, 0]
            for m in range(1, n_machines):
                C[j_idx, m] = max(C[j_idx, m-1], C[j_idx-1, m]) + processing_times[job, m]
                
    makespan = C[-1, -1]
    return makespan, C


def _delta_makespan(perm, C, processing_times, i, j):
    """
    Calculate the new makespan after moving job at index i to index j in perm.
    This is an approximation/optimization: we recalculate the completion times 
    for the segment from min(i,j) to the end.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    # Determine the range of indices that are affected
    # If we move job i to j, the jobs between i and j shift.
    # The completion times for jobs before min(i, j) remain unchanged.
    # The job at i is removed.
    # The jobs between i and j (exclusive) shift by 1.
    # The job at i is inserted at j.
    # All jobs from min(i, j) to end are affected.
    
    start_idx = min(i, j)
    
    # We need to construct the new permutation segment to compute C
    # However, for speed, we can just simulate the flow shop for the affected part.
    # But we need the C values from the previous unaffected job.
    
    # Let's create a new C matrix for the affected part, or just a list of completion times for the last job on each machine?
    # No, we need the full row for each affected job to pass to the next.
    
    # To avoid creating a new full C matrix, we can just compute the makespan for the suffix.
    # We need the "state" at start_idx - 1.
    # State is the vector of completion times of the job at start_idx - 1 on each machine.
    
    if start_idx == 0:
        prev_state = np.zeros(n_machines)
    else:
        prev_state = C[start_idx - 1].copy()
        
    # Create the new permutation order for the suffix
    # Original suffix: perm[start_idx:]
    # Move job at i to j.
    # If i < j:
    #   Original: [ ... , p_i, p_{i+1}, ..., p_j, ... ]
    #   New:      [ ... , p_{i+1}, ..., p_j, p_i, ... ]
    #   The job p_i moves from i to j.
    #   The jobs i+1 to j shift left by 1.
    # If i > j:
    #   Original: [ ... , p_j, ..., p_i, ... ]
    #   New:      [ ... , p_j, p_i, ..., p_{i-1}, ... ]
    #   The job p_i moves to j.
    #   The jobs j to i-1 shift right by 1.
    
    # Let's just build the new suffix permutation list and compute C for it.
    # This is O(N^2) in the worst case if we copy lists, but N is usually small enough (<100) that this is fine.
    # For larger N, we might need to be more careful, but let's try this first.
    
    suffix_start = start_idx
    suffix_len = n_jobs - suffix_start
    
    # Get the jobs in the suffix
    suffix_jobs = perm[suffix_start:]
    
    # Perform the move on this suffix list
    # Indices i and j are relative to the whole perm, so relative to suffix:
    rel_i = i - suffix_start
    rel_j = j - suffix_start
    
    # Remove job at rel_i
    job_to_move = suffix_jobs.pop(rel_i)
    # Insert at rel_j
    # Note: if rel_i < rel_j, the pop shifts elements left, so the target index rel_j is now effectively rel_j-1?
    # No, we want the job to be at position j in the *original* indexing?
    # The problem definition for 'move' is usually: remove from i, insert at j.
    # If i < j, inserting at j in the *new* list (after removal) means it ends up at index j in the *new* list?
    # Or does it mean it ends up at index j in the *original* list position?
    # Standard interpretation: The job is removed from i. The list shrinks. Then it is inserted at index j.
    # If i < j, the insertion index j in the *new* list corresponds to index j in the new list.
    # Let's stick to the standard Python list semantics: pop(i), then insert(j, item).
    
    suffix_jobs.insert(rel_j, job_to_move)
    
    # Now compute makespan for this suffix, starting from prev_state
    # prev_state is the completion times of the job *before* suffix_start.
    
    # We need to simulate the flow shop for the jobs in suffix_jobs.
    # current_state[m] = completion time of the *previous* job in the sequence on machine m.
    # Initially, current_state = prev_state.
    
    current_state = prev_state.copy()
    
    for job in suffix_jobs:
        # Process job on all machines
        new_state = np.zeros(n_machines)
        # Machine 0
        new_state[0] = current_state[0] + processing_times[job, 0]
        current_state[0] = new_state[0] # Update for next machine? No, we need the completion on m-1 for the current job.
        
        # Let's do it properly:
        # C_job_m = max(C_job_{m-1}, C_prev_job_m) + P_job_m
        
        # We need to track C_job_{m-1} as we go.
        c_prev_m = 0.0 # Completion of current job on previous machine
        for m in range(n_machines):
            if m == 0:
                c_curr = current_state[0] + processing_times[job, 0]
            else:
                c_curr = max(c_prev_m, current_state[m]) + processing_times[job, m]
            
            # Update current_state[m] to be the completion of this job on machine m
            # This will be the "prev job" for the next job in the sequence
            current_state[m] = c_curr
            c_prev_m = c_curr
            
    return current_state[-1]


def _neh_initialization(processing_times):
    """
    NEH heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return [], np.zeros((0, n_machines))
    
    total_processing = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_processing)
    
    current_perm = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm.copy()
            new_perm.insert(pos, job)
            # For initialization, we can just use the fast makespan calc
            # Or we can maintain C, but for init, simple is fine.
            ms, _ = _compute_makespan_matrix(new_perm, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm.insert(best_pos, job)
        
    # Compute final C
    makespan, C = _compute_makespan_matrix(current_perm, processing_times)
    return current_perm, C


def _tabu_search(initial_perm, C, processing_times, time_limit=10.0):
    """
    Tabu search with delta evaluation.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _compute_makespan_matrix(current_perm, processing_times)[0]
    # Re-compute C to be sure
    current_makespan, current_C = _compute_makespan_matrix(current_perm, processing_times)
    
    best_perm = list(current_perm)
    best_makespan = current_makespan
    best_C = current_C.copy()
    
    tabu_list = {}
    tabu_tenure = 10
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50
    
    # Pre-compute some info?
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        best_neighbor_C = None
        
        # Generate moves
        # We will try all adjacent swaps and a sample of insertions.
        # For delta evaluation, we need to pass the current C.
        
        moves_to_try = []
        
        # Adjacent swaps
        for i in range(n_jobs - 1):
            moves_to_try.append(('swap', i, i + 1))
            
        # Insertions (Or-opt style or random)
        # Sample 50 random insertions
        for _ in range(50):
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                moves_to_try.append(('move', i, j))
                
        for move in moves_to_try:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                move_key = ('swap', min(i, j), max(i, j))
            else:
                i, j = move[1], move[2]
                if i == j:
                    continue
                move_key = ('move', i, j)
                
            # Check tabu
            if move_key in tabu_list and tabu_list[move_key] > iteration:
                # Aspiration check later
                pass
                
            # Calculate new makespan using delta
            # For swap, it's a move of i to j (or j to i).
            # Let's treat swap as two moves or just use the delta function for 'move'.
            # If we swap i and i+1, it's equivalent to moving i to i+1 or i+1 to i.
            # Let's just use the 'move' delta logic for swaps too?
            # Swap i, j:
            #   Remove i, insert at j (if i<j) or remove j, insert at i (if i>j)?
            #   Actually, a swap is not exactly a move.
            #   Let's just use full re-evaluation for swaps if N is small, or optimize.
            #   Given the delta function works for 'move', we can implement a specific delta for swap.
            #   Or, simpler: just use the delta function for 'move' and skip swaps?
            #   Swaps are a good neighborhood.
            #   Let's implement a specific delta for swap.
            
            if move[0] == 'swap':
                # Swap i and j (adjacent)
                # The affected segment is from i to j (inclusive) and potentially after?
                # Actually, swapping two adjacent jobs only affects the completion times of these two jobs and subsequent jobs.
                # So start_idx = i.
                
                # We can use the same logic as move, but we need to handle the permutation change correctly.
                # Let's just reuse the _delta_makespan logic by converting swap to a move sequence?
                # No, let's just write a specific delta for swap.
                
                # For now, to keep it simple and robust, let's use the generic move delta for 'move'
                # and for 'swap', we can just do a full re-evaluation if n_jobs < 50, 
                # or use the move delta by simulating the swap as two moves? No, that's complex.
                
                # Let's just use the _delta_makespan function for 'move'
                # and for 'swap', we will just do a quick full re-evaluation using _compute_makespan_matrix
                # if the problem is small, or use a specialized delta.
                
                # Actually, the _delta_makespan function I wrote above takes i and j for a 'move'.
                # A swap of i and j is not a move.
                # Let's just use full re-evaluation for swaps in this version to ensure correctness,
                # and rely on the high frequency of 'move' operations for speed.
                # Or, better: implement a specific delta for swap.
                
                # Swap delta:
                # The jobs at i and j are swapped.
                # The completion times for jobs < i are unchanged.
                # The job at i becomes job_j, and job at j becomes job_i.
                # The jobs between i and j (if any) are unchanged in order but their start times change.
                # So we need to recalculate from i to end.
                
                # Let's just use the _compute_makespan_matrix for swaps for now, 
                # and _delta_makespan for moves.
                # If n_jobs is large, swaps might be expensive. 
                # But adjacent swaps are local.
                
                # Let's try to use the delta logic for swap by treating it as:
                # 1. Move i to j
                # 2. Move old j (now at j-1 or j+1) to i?
                # Too complex.
                
                # Let's just use full re-eval for swaps.
                new_perm = current_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_makespan, new_C = _compute_makespan_matrix(new_perm, processing_times)
                
            else:
                i, j = move[1], move[2]
                # Use delta evaluation
                new_makespan = _delta_makespan(current_perm, current_C, processing_times, i, j)
                
                # If we accept this move, we need the new C.
                # We don't have it yet. We only have the makespan.
                # To maintain C for the next iteration, we need to update it.
                # We can update C by recalculating the affected segment.
                # Let's do that only if this move is the best neighbor.
                # For now, just store the makespan.
                
                # Wait, if we don't have new_C, we can't use delta evaluation for the NEXT iteration if we accept this move.
                # So we must compute new_C if we accept.
                # But we don't know if we accept until we check all neighbors.
                # So we should store the move, and if it's the best, we compute the new C.
                
                # To avoid computing C for every neighbor, we only compute it for the best neighbor.
                # But we need the makespan for all neighbors.
                # The delta function gives us the makespan.
                # So we store (move, makespan).
                # If it's the best, we compute the new C.
                
                pass # Logic handled below
                
            # Check tabu and aspiration
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            is_aspiration = new_makespan < best_makespan
            
            if is_tabu and not is_aspiration:
                continue
            
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_move = move
                # Store the move to reconstruct C later
                best_neighbor_perm_candidate = None # We'll reconstruct from move
                
        if best_neighbor_move is None:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            # Perturb
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                new_perm = current_perm.copy()
                job_to_move = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
                current_perm = new_perm
                current_makespan, current_C = _compute_makespan_matrix(current_perm, processing_times)
        else:
            no_improve_count = 0
            # Update tabu list
            # Reconstruct move_key
            if best_neighbor_move[0] == 'swap':
                i, j = best_neighbor_move[1], best_neighbor_move[2]
                move_key = ('swap', min(i, j), max(i, j))
            else:
                i, j = best_neighbor_move[1], best_neighbor_move[2]
                move_key = ('move', i, j)
                
            tabu_list[move_key] = iteration + tabu_tenure
            
            # Clean up tabu list
            for key in list(tabu_list.keys()):
                if tabu_list[key] <= iteration:
                    del tabu_list[key]
                    
            # Apply the best move to current_perm and current_C
            # We need to generate the new_perm and compute new_C.
            if best_neighbor_move[0] == 'swap':
                i, j = best_neighbor_move[1], best_neighbor_move[2]
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            else:
                i, j = best_neighbor_move[1], best_neighbor_move[2]
                job_to_move = current_perm.pop(i)
                if i < j:
                    j -= 1
                current_perm.insert(j, job_to_move)
                
            current_makespan, current_C = _compute_makespan_matrix(current_perm, processing_times)
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
                best_C = current_C.copy()
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: NEH
    initial_perm, initial_C = _neh_initialization(processing_times)
    
    # Refine with tabu search
    n_jobs = len(processing_times)
    if n_jobs <= 20:
        time_limit = 15.0
    elif n_jobs <= 50:
        time_limit = 10.0
    else:
        time_limit = 5.0
        
    final_perm = _tabu_search(initial_perm, initial_C, processing_times, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
