"""
Strategy: Minimize the sum of powers of discrepancies (e.g., sum of squares or 4th powers) 
to achieve a balanced distribution of errors across all arithmetic progressions.
This avoids the "rugged" landscape of minimizing the max directly, which often gets stuck
in local optima where one progression is small but many are large.
After the balancing phase, a short local search phase minimizes the max discrepancy.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_data(n):
    """
    Pre-compute data for all arithmetic progressions.
    Returns:
    - progressions: list of arrays of indices
    - index_to_prog_pos: list of lists of (prog_id, pos_in_prog)
    """
    progressions = []
    index_to_prog_pos = [[] for _ in range(n)]
    
    pid = 0
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                break
            indices = np.arange(r, n, step, dtype=np.int32)
            if len(indices) == 0:
                continue
            progressions.append(indices)
            for pos, idx in enumerate(indices):
                index_to_prog_pos[idx].append((pid, pos))
            pid += 1
            
    return progressions, index_to_prog_pos

def _compute_objective(signs, progressions, index_to_prog_pos, power=2):
    """
    Compute sum of (abs(sum)^power) over all progressions.
    Used as the smooth objective function for the main optimization.
    """
    arr = np.array(signs, dtype=np.int64)
    total_obj = 0
    for indices in progressions:
        subseq = arr[indices]
        sums = np.cumsum(subseq)
        # We care about the max absolute sum in the progression? 
        # No, the task asks for the max absolute signed sum on ANY arithmetic progression.
        # Wait, the task says: "Minimize the maximum absolute signed sum on ANY arithmetic progression".
        # This means for a specific progression (start, step), we take the sum of the *entire* progression?
        # Or the max over all sub-progressions?
        # "contained in [1,256], any start, positive step and length including one."
        # Usually "discrepancy on an AP" means the sum of the signs in that AP.
        # If the AP is defined by start, step, and length, the sum is fixed for that specific AP.
        # The "maximum" is taken over ALL such APs.
        # So the objective is: max_{all APs} |sum(signs in AP)|.
        
        # My previous interpretation of "prefix sums" was slightly off if I was looking at max abs prefix sum.
        # Let's re-read carefully: "maximum absolute signed sum on ANY arithmetic progression".
        # An arithmetic progression is a set of numbers {a, a+d, a+2d, ...}.
        # The sum is S = sum_{k} signs[a+kd].
        # We want to minimize max_{a,d,L} |S|.
        
        # So for each specific progression instance (defined by start, step, length), 
        # we compute the sum.
        # However, usually "discrepancy" in this context (Erdos) refers to the max over all *sub-sequences* 
        # or just the full sum? 
        # "any start, positive step and length including one".
        # This implies we consider ALL possible APs of ANY length.
        # So if we have start=1, step=2, we have APs of length 1, 2, 3...
        # Each of these is a distinct "arithmetic progression" in the set of all such.
        # So yes, we need to consider all prefixes of the infinite APs, truncated to [1, n].
        
        # So for a fixed (start, step), we have a sequence s_1, s_2, ..., s_k.
        # The sums are S_1 = s_1, S_2 = s_1+s_2, ..., S_k = s_1+...+s_k.
        # The discrepancy for this (start, step) is max(|S_1|, ..., |S_k|).
        # The global discrepancy is max over all (start, step) of that value.
        
        # So the "smooth" objective should be sum over all (start, step) of (max_prefix_sum^power).
        # Or even sum over all individual prefix sums? 
        # Sum of squares of all individual prefix sums is a very smooth proxy for the max.
        
        abs_sums = np.abs(sums)
        # Using sum of squares of all prefix sums encourages keeping ALL prefix sums small.
        total_obj += np.sum(abs_sums ** power)
        
    return total_obj

def _compute_max_discrepancy(signs, progressions):
    """
    Compute the actual metric: max absolute signed sum on ANY arithmetic progression.
    """
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    for indices in progressions:
        subseq = arr[indices]
        cumsums = np.cumsum(subseq)
        max_abs = np.max(np.abs(cumsums))
        if max_abs > max_disc:
            max_disc = max_abs
    return max_disc

def construct():
    n = 256
    deadline = time.time() + 55  # Safety margin for 90s limit (evaluator kills at 90s, but task says 60s finish? "Each evaluation must finish within 60 s". Let's aim for 50s to be safe).
    
    # Pre-compute AP data
    progressions, index_to_prog_pos = _build_ap_data(n)
    num_progs = len(progressions)
    
    # We will use multiple restarts.
    # Phase 1: Minimize Sum of Squares (or 4th powers) of all prefix sums.
    # Phase 2: Local search on Max Discrepancy.
    
    best_signs = None
    best_max_disc = float('inf')
    
    rng = random.Random(42)
    
    # Number of restarts depends on time.
    # Let's estimate time per restart.
    
    for restart in range(20): # Cap restarts, break on time
        if time.time() > deadline - 10:
            break
            
        # Initialize random signs
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        signs_arr = np.array(signs, dtype=np.int64)
        
        # Calculate initial smooth objective
        # To speed up updates, we maintain the sum of squares of prefix sums for each progression.
        # obj = sum_{pid} sum_{pos} (prefix_sum[pid][pos]^2)
        
        prefix_sums = []
        prog_objs = np.zeros(num_progs, dtype=np.float64)
        
        for pid, indices in enumerate(progressions):
            subseq = signs_arr[indices]
            cs = np.cumsum(subseq)
            prefix_sums.append(cs)
            prog_objs[pid] = np.sum(cs ** 2)
            
        current_smooth_obj = np.sum(prog_objs)
        
        # Simulated Annealing on Smooth Objective
        # We want to minimize current_smooth_obj.
        T = 10.0
        alpha = 0.99
        iterations = 0
        max_iter = 5000 # Fixed iterations per restart for smooth phase
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            i = rng.randint(0, n - 1)
            old_val = signs[i]
            new_val = -old_val
            delta_val = new_val - old_val
            
            affected = index_to_prog_pos[i]
            
            old_obj_delta = 0.0
            old_prefix_slices = []
            
            for pid, pos in affected:
                ps = prefix_sums[pid]
                old_ps_suffix = ps[pos:].copy()
                old_obj = prog_objs[pid]
                old_prefix_slices.append((pid, pos, old_ps_suffix, old_obj))
                
                # Update prefix sums
                ps[pos:] += delta_val
                
                # Update objective for this progression
                # new_obj = sum(ps^2)
                # We can update incrementally? 
                # sum((old+delta)^2) = sum(old^2) + 2*delta*sum(old) + len*delta^2
                # But we don't have sum(old) stored easily for the suffix.
                # Given small n, just recompute.
                new_obj = np.sum(ps ** 2)
                old_obj_delta += (new_obj - old_obj)
                prog_objs[pid] = new_obj
                
            new_smooth_obj = current_smooth_obj + old_obj_delta
            
            # Acceptance
            if new_smooth_obj < current_smooth_obj:
                signs[i] = new_val
                current_smooth_obj = new_smooth_obj
            else:
                delta_obj = new_smooth_obj - current_smooth_obj
                if rng.random() < np.exp(-delta_obj / T):
                    signs[i] = new_val
                    current_smooth_obj = new_smooth_obj
                else:
                    # Revert
                    for pid, pos, old_ps_suffix, old_obj in old_prefix_slices:
                        prefix_sums[pid][pos:] = old_ps_suffix
                        prog_objs[pid] = old_obj
                        
            T *= alpha
            if T < 0.01:
                T = 0.01
            iterations += 1
            
        # End of Smooth Phase for this restart.
        
        # Phase 2: Local Search on Max Discrepancy
        # Start from the state produced by smooth phase.
        current_max_disc = _compute_max_discrepancy(signs, progressions)
        
        # Simple hill climbing / local search to reduce max disc
        # We need to be careful with time here.
        ls_iterations = 0
        max_ls_iter = 2000
        
        # Pre-compute prefix sums again for the current state to speed up max disc calc?
        # _compute_max_discrepancy is O(P * L) ~ 32k * avg_len. 
        # It's fast enough (~ms).
        
        for _ in range(max_ls_iter):
            if time.time() > deadline - 2:
                break
                
            i = rng.randint(0, n - 1)
            old_val = signs[i]
            new_val = -old_val
            
            # Try flip
            signs[i] = new_val
            new_max_disc = _compute_max_discrepancy(signs, progressions)
            
            if new_max_disc < current_max_disc:
                current_max_disc = new_max_disc
            else:
                # Revert
                signs[i] = old_val
            ls_iterations += 1
            
        # Check if this restart improved the global best
        if current_max_disc < best_max_disc:
            best_max_disc = current_max_disc
            best_signs = list(signs)
            
        # Optional: If we have time, do a final polish on the best_signs found so far?
        # Or just continue with next restart.
        
    if best_signs is None:
        # Fallback
        best_signs = [1] * n
        
    return best_signs
# EVOLVE-BLOCK-END
