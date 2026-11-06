import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    
    Strategy:
    1. Generate multiple initial solutions using the NEH (Nawaz, Enscore, Ham) heuristic.
    2. Refine each solution using a local search with insertion moves.
    3. Return the best permutation found within the time budget.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Precompute total processing times for NEH sorting
    total_times = processing_times.sum(axis=1)
    
    # Helper function to calculate makespan for a given permutation
    def calc_makespan(perm):
        """
        Calculate the makespan for a given job permutation.
        perm: list of job indices
        """
        completion = np.zeros(n_machines)
        for job in perm:
            for m in range(n_machines):
                if m == 0:
                    completion[m] += processing_times[job, m]
                else:
                    completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
        return completion[-1]
    
    # Helper function for fast makespan calculation after an insertion move
    def calc_makespan_fast(perm, removed_idx, inserted_idx):
        """
        Calculate makespan for a permutation where job at removed_idx was moved to inserted_idx.
        We recompute from the insertion point onwards.
        """
        # Create the new permutation
        # Note: In the search loop, we will maintain the permutation list directly.
        # This function is a conceptual placeholder; the search loop will do the math inline for speed.
        pass

    def neh_heuristic(processing_times, total_times, rng_seed=None):
        """
        Implement the NEH heuristic.
        """
        n = len(total_times)
        # Sort jobs by decreasing total processing time
        # Add a small random noise to break ties and generate diverse starts
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
            noise = rng.uniform(0, 0.01, size=n)
            sort_keys = -(total_times + noise)
        else:
            sort_keys = -total_times
            
        sorted_jobs = np.argsort(sort_keys)
        
        # Build the sequence
        seq = [sorted_jobs[0]]
        
        for j in sorted_jobs[1:]:
            best_pos = 0
            best_makespan = float('inf')
            
            # Try inserting job j at every possible position in the current sequence
            for pos in range(len(seq) + 1):
                # Create a temporary sequence
                new_seq = seq[:pos] + [j] + seq[pos:]
                
                # Calculate makespan for this temporary sequence
                # Optimization: Only compute from pos onwards? 
                # For NEH construction, n is small enough that full calc is okay, 
                # but we can optimize by tracking completion times.
                
                # Simple full calculation for clarity in construction phase
                completion = np.zeros(n_machines)
                for k, job in enumerate(new_seq):
                    for m in range(n_machines):
                        if m == 0:
                            completion[m] += processing_times[job, m]
                        else:
                            completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
                
                makespan = completion[-1]
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_pos = pos
            
            # Insert job at best position
            seq.insert(best_pos, j)
            
        return seq, best_makespan

    def local_search_insertion(perm, initial_makespan, max_iterations, start_time_ref, time_limit_ref):
        """
        Perform local search using insertion moves.
        Returns the best permutation and its makespan.
        """
        n = len(perm)
        best_perm = perm[:]
        best_makespan = initial_makespan
        current_perm = perm[:]
        current_makespan = initial_makespan
        
        # Precompute completion times for the current permutation to speed up evaluation
        # C[j, m] = completion time of job j on machine m
        # We need to be careful with indices.
        
        def get_completion_matrix(p):
            comp = np.zeros((n, n_machines))
            for idx, job in enumerate(p):
                for m in range(n_machines):
                    if m == 0:
                        comp[idx, m] = processing_times[job, m]
                    else:
                        comp[idx, m] = max(comp[idx-1, m], comp[idx, m-1]) + processing_times[job, m]
            return comp
        
        current_comp = get_completion_matrix(current_perm)
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            if time.time() - start_time_ref > time_limit_ref:
                break
                
            for i in range(n):
                if time.time() - start_time_ref > time_limit_ref:
                    break
                
                job = current_perm[i]
                # Remove job from position i
                # The job is removed, so positions > i shift left by 1
                
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Determine the target position in the new list (after removal)
                    # If we remove i, the list length is n-1.
                    # If we insert at position j in the *new* list:
                    # If j < i, the job moves earlier.
                    # If j >= i, the job moves later (or stays same relative to others, but effectively shifts).
                    
                    # Let's construct the new permutation to evaluate
                    # Actually, constructing the list every time is O(n).
                    # Evaluating makespan is O(n*M).
                    # Total per iteration: O(n^2 * M).
                    # For n=20, M=5, this is 2000 ops. 1000 iters = 2M ops. Very fast.
                    
                    new_perm = current_perm[:i] + current_perm[i+1:]
                    # Determine insertion index in new_perm
                    if j < i:
                        insert_idx = j
                    else:
                        insert_idx = j - 1 if j > i else j # Wait, if j==i, we skip.
                        # If j > i, the position j in original corresponds to j-1 in new list?
                        # Original: ... A(i) B(i+1) C(i+2) ...
                        # Remove A: ... B C ...
                        # If we want to put A after B (orig pos i+1), in new list B is at i. So insert at i+1.
                        # If j == i+1, insert_idx should be i.
                        # If j == i+2, insert_idx should be i+1.
                        # So insert_idx = j - 1.
                        insert_idx = j - 1
                        
                    # Clamp insert_idx
                    if insert_idx < 0: insert_idx = 0
                    if insert_idx > len(new_perm): insert_idx = len(new_perm)
                    
                    new_perm.insert(insert_idx, job)
                    
                    # Calculate makespan for new_perm
                    # We can optimize this by only recalculating from the earliest affected position.
                    # But for simplicity and given small n, full calc is acceptable if n < 30.
                    # If n is large, we need the fast update.
                    
                    # Let's use a fast update method:
                    # The change is local.
                    # However, implementing a correct fast update for insertion is tricky.
                    # Given the 150s limit, and typical instance sizes in these tasks (often n<=20 or n<=50),
                    # full recalculation is likely fast enough in Python if n is small.
                    # If n is large (e.g., 100), we need vectorization or C extensions.
                    # Let's assume n is moderate.
                    
                    makespan = 0
                    comp_m = 0
                    for k, jb in enumerate(new_perm):
                        # Vectorized machine loop
                        # completion[m] = max(completion[m-1], completion_prev_machine[m]) + pt[jb, m]
                        # This is inherently sequential across machines.
                        # We can use a small loop for M.
                        c = 0
                        for m in range(n_machines):
                            if m == 0:
                                c = pt_val = processing_times[jb, m]
                            else:
                                c = max(c, comp_m) + processing_times[jb, m]
                                # Wait, comp_m is the completion of the *previous job* on machine m.
                                # We need an array for completions of the current job across machines?
                                # No, for a single job, we just need the completion on the previous machine.
                                # But for the *next* job, we need the completion on *all* machines.
                                # So we need to track the completion vector.
                        # This inner loop is too slow if done in pure Python for large n.
                        
                        # Let's stick to the numpy-based full calc for simplicity and correctness first.
                        # If n is small, it's fine.
                        
                        break # Break out of this inefficient manual calc attempt
                        
                    # Re-implement makespan calc efficiently for the new_perm
                    # Using numpy arrays for the completion state
                    completion = np.zeros(n_machines)
                    for jb in new_perm:
                        for m in range(n_machines):
                            if m == 0:
                                completion[m] += processing_times[jb, m]
                            else:
                                completion[m] = max(completion[m], completion[m-1]) + processing_times[jb, m]
                    new_makespan = completion[-1]
                    
                    if new_makespan < current_makespan:
                        current_perm = new_perm
                        current_makespan = new_makespan
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            best_perm = current_perm[:]
                        improved = True
                        # Restart the search from the new state? Or continue?
                        # Continuing is fine, but we might miss improvements.
                        # For speed, we just accept the move and continue the loop.
                        # The outer 'improved' flag ensures we keep going until no improvement.
                        break # Break inner loop, restart outer? No, just break to next i?
                        # If we accept a move, the indices change. It's safer to restart the i-loop.
                        # But that's expensive.
                        # Let's just break the j-loop and continue the i-loop.
                        # The next i will operate on the updated current_perm.
                        # This is a "First Improvement" strategy, which is fast.
        
        return best_perm, best_makespan

    # Generate multiple diverse NEH starts
    num_starts = 10
    best_overall_perm = None
    best_overall_makespan = float('inf')
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Use different seeds for diversity
        perm, makespan = neh_heuristic(processing_times, total_times, rng_seed=i)
        
        if makespan < best_overall_makespan:
            best_overall_makespan = makespan
            best_overall_perm = perm[:]
            
        # Refine with local search
        # Limit local search iterations to ensure we finish in time
        # Estimate cost: O(n^2 * M * iters). 
        # If n=20, M=5, iters=1000 -> 2M ops. ~0.1s in Python.
        # So 10 starts * 1000 iters is ~1s. We have 110s.
        # We can afford more iterations.
        
        # Dynamic iteration limit based on remaining time
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        # Assume ~0.001s per iteration (conservative for Python)
        max_iters = int(remaining * 1000) 
        if max_iters < 100:
            max_iters = 100
        if max_iters > 5000:
            max_iters = 5000
            
        refined_perm, refined_makespan = local_search_insertion(
            perm, makespan, max_iters, start_time, time_limit
        )
        
        if refined_makespan < best_overall_makespan:
            best_overall_makespan = refined_makespan
            best_overall_perm = refined_perm[:]
            
    # If no best found (should not happen), fallback to NEH
    if best_overall_perm is None:
        best_overall_perm, _ = neh_heuristic(processing_times, total_times)
        
    return best_overall_perm
