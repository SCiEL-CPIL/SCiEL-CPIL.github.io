import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        # perm is a list of job indices
        n = len(perm)
        if n == 0:
            return 0.0
        
        # Get processing times for all jobs in the permutation
        job_indices = np.array(perm, dtype=np.int64)
        job_times = processing_times[job_indices]  # shape: (n_jobs, n_machines)
        
        # Compute completion times iteratively (can't fully vectorize due to dependencies)
        completion = np.zeros(n_machines)
        for j in range(n):
            jt = job_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
                completion[m] = st + jt[m]
        return completion[-1]

    # 1. V-shaped heuristic construction
    # In flow shop, V-shaped heuristic tends to work well:
    # Sort jobs by total processing time, place largest first and smallest last
    total_times = processing_times.sum(axis=1)
    sorted_indices = np.argsort(total_times)
    
    # Create V-shaped sequence
    seq = [0] * n_jobs
    left = 0
    right = n_jobs - 1
    
    # Place jobs in V-shape: largest in middle, smallest at ends
    # Actually, standard V-shape: sort by total time, place largest first, then smallest, then 2nd largest, etc.
    # Or: place largest in center, expand outward
    # Let's use: sort descending, place first in center, second to left, third to right, etc.
    
    # Alternative simpler V-shape: 
    # Sort by total processing time descending
    # Place first job at position 0
    # Place second job at position n-1
    # Place third job at position 1
    # Place fourth job at position n-2
    # etc.
    
    sorted_desc = np.argsort(-total_times)
    seq = [0] * n_jobs
    left = 0
    right = n_jobs - 1
    place_left = True
    
    for idx in sorted_desc:
        if place_left:
            seq[left] = idx
            left += 1
        else:
            seq[right] = idx
            right -= 1
        place_left = not place_left
    
    best_seq = list(seq)
    best_ms = calc_makespan(best_seq)
    
    # 2. Tabu Search
    # Dynamic tabu list to prevent cycling
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    tabu_tenure = 10  # base tenure
    current_seq = list(seq)
    current_ms = best_ms
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        improved = False
        n = len(current_seq)
        
        # Adaptive tabu tenure
        if no_improve_count > 20:
            tabu_tenure = min(50, tabu_tenure + 2)
        elif no_improve_count < 10:
            tabu_tenure = max(5, tabu_tenure - 1)
        
        # Try all swap moves
        best_move = None
        best_move_ms = current_ms
        best_move_i = -1
        best_move_j = -1
        
        for i in range(n - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                # Check if this move is tabu
                move_key = (i, j)
                if move_key in tabu_list and tabu_list[move_key] > iteration:
                    continue
                
                # Try the swap
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                ms = calc_makespan(current_seq)
                
                # Accept if better, or if not tabu and we need to escape local minimum
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_move = (i, j)
                
                # Revert swap
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        
        if best_move is not None:
            i, j = best_move
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            
            # Add to tabu list
            tabu_list[(i, j)] = iteration + tabu_tenure
            tabu_list[(j, i)] = iteration + tabu_tenure  # symmetric
            
            if best_move_ms < current_ms:
                current_ms = best_move_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = list(current_seq)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Non-improving move (tabu search allows this to escape local minima)
                no_improve_count += 1
                current_ms = best_move_ms
            
            improved = True
        else:
            # No non-tabu moves available, clear tabu list and allow any move
            tabu_list.clear()
            break
        
        iteration += 1
        
        # Clean up old tabu entries
        if iteration % 100 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
    
    # 3. Final local search polish
    if time.time() < deadline - 2:
        seq = list(best_seq)
        ms = best_ms
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(seq)
            for i in range(n - 1):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    new_ms = calc_makespan(seq)
                    if new_ms < ms:
                        ms = new_ms
                        best_ms = ms
                        best_seq = list(seq)
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
    
    return best_seq
