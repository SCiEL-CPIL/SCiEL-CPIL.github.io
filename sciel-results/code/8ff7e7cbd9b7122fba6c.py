"""Flow-shop permutation via multi-start heuristics + fast insertion/reversal local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(pt_list, perm):
    """
    Fast makespan calculation using pure Python lists.
    pt_list: list of lists, shape [n_jobs][n_machines].
    perm: list of job indices.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0 or n_machines == 0:
        return 0.0

    # prev[m] is the completion time of the previous job on machine m
    prev = [0.0] * n_machines

    for job in perm:
        row = pt_list[job]
        # curr[m] is the completion time of the current job on machine m
        # curr[0] = prev[0] + row[0]
        # curr[m] = max(prev[m], curr[m-1]) + row[m]
        curr_0 = prev[0] + row[0]
        if n_machines == 1:
            prev[0] = curr_0
            continue
        
        curr_prev = curr_0
        for m in range(1, n_machines):
            # max(prev[m], curr_prev)
            if prev[m] > curr_prev:
                val = prev[m]
            else:
                val = curr_prev
            curr_m = val + row[m]
            curr_prev = curr_m
            # We only need to store the final result for the next job, 
            # but we need the whole row 'prev' to be updated for the next job.
            # So we must update prev[m] as we go? No, prev[m] is from the *previous* job.
            # We need to construct the new 'prev' array.
            # To save time, we can just keep a single 'curr' array and swap references?
            # Or just update in place if we are careful?
            # Let's use a temporary list for curr to avoid aliasing issues, 
            # or just update a new list.
            pass
        
        # The loop above was inefficient. Let's do it properly.
        # We need a new prev array.
        # Optimization: Reuse lists.
        pass

    # Let's rewrite _makespan_fast to be truly fast.
    pass

def _makespan_fast_optimized(pt_list, perm):
    """
    Optimized makespan calculation.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0 or n_machines == 0:
        return 0.0

    if n_machines == 1:
        return sum(pt_list[job][0] for job in perm)

    # prev[m] holds completion time of the last processed job on machine m
    prev = [0.0] * n_machines
    
    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
            # prev[m] is C[j-1, m]
            # curr[m-1] is C[j, m-1]
            if prev[m] > curr[m-1]:
                curr[m] = prev[m] + row[m]
            else:
                curr[m] = curr[m-1] + row[m]
        prev = curr
        
    return prev[-1]


def _neh_initial(pt_list, n_jobs, n_machines, sort_key_func):
    """
    NEH heuristic with a specific sorting key.
    """
    # Sort jobs based on key
    jobs = list(range(n_jobs))
    jobs.sort(key=lambda j: sort_key_func(j), reverse=True)
    
    perm = [jobs[0]]
    
    for i in range(1, n_jobs):
        job = jobs[i]
        best_pos = 0
        best_ms = float('inf')
        # Try inserting job at every position
        for pos in range(len(perm) + 1):
            # To avoid creating many lists, we can compute makespan incrementally?
            # For NEH, N is usually small enough that list insertion + makespan is fine.
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast_optimized(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
        
    return perm


def _palmer_scores(pt_list, n_jobs, n_machines):
    """Calculate Palmer's dispatching rule scores."""
    scores = [0.0] * n_jobs
    # Palmer's rule: sum_{m=1}^{M} (M+1-2m) * p_{jm}
    # Or: sum_{m=1}^{M/2} (M+1-2m)p_{jm} - sum_{m=M/2+1}^{M} (M+1-2m)p_{jm}
    # Simplified: (M+1) * sum(p) - 2 * sum(m * p)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n_machines):
            s += (n_machines + 1 - 2 * m) * pt_list[j][m]
        scores[j] = s
    return scores


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 5 seconds margin before 150s limit
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Convert to list of lists for faster access in local search
    pt_list = pt.tolist()
    
    # --- Step 1: Generate Initial Candidates ---
    
    candidates = []
    
    # 1. NEH (Total processing time)
    totals = [sum(pt_list[j]) for j in range(n_jobs)]
    candidates.append(_neh_initial(pt_list, n_jobs, n_machines, lambda j: totals[j]))
    
    # 2. NEH (Mean processing time)
    means = [sum(pt_list[j]) / n_machines for j in range(n_jobs)]
    candidates.append(_neh_initial(pt_list, n_jobs, n_machines, lambda j: means[j]))
    
    # 3. Palmer's Rule (sorted descending by score)
    palmer_scores = _palmer_scores(pt_list, n_jobs, n_machines)
    palmer_order = list(range(n_jobs))
    palmer_order.sort(key=lambda j: palmer_scores[j], reverse=True)
    candidates.append(palmer_order)
    
    # 4. Sort by first machine time
    candidates.append(sorted(range(n_jobs), key=lambda j: pt_list[j][0], reverse=True))
    
    # 5. Sort by last machine time
    candidates.append(sorted(range(n_jobs), key=lambda j: pt_list[j][-1], reverse=True))
    
    # 6. Random candidates
    rng = np.random.RandomState(42)
    for _ in range(10):
        candidates.append(rng.permutation(n_jobs).tolist())

    # --- Step 2: Local Search (VND-like) ---
    
    best_perm = None
    best_makespan = float('inf')
    
    def _local_search(perm):
        """
        Local search using Insertion and Reversal moves.
        First Improvement strategy.
        """
        current = perm[:]
        current_ms = _makespan_fast_optimized(pt_list, current)
        
        # We alternate between Insertion and Reversal neighborhoods
        # Or just do a combined loop.
        
        improved = True
        while improved:
            improved = False
            
            # Check deadline
            if time.time() > deadline:
                break
            
            # --- Neighborhood 1: Insertion ---
            # Remove job i, insert at k
            n = len(current)
            for i in range(n):
                if time.time() > deadline:
                    break
                job = current[i]
                rest = current[:i] + current[i+1:]
                for k in range(len(rest) + 1):
                    if time.time() > deadline:
                        break
                    # Skip if inserting at original position (i or i-1 in rest)
                    # If i > k, original pos is i-1 in rest.
                    # If i <= k, original pos is i in rest.
                    if k == i or (k == i - 1 and i > 0):
                        continue
                    # Also skip if k == i (when i < len(rest)) which is same as original
                    # Actually, if we remove i, the list is length n-1.
                    # Original position was i.
                    # If we insert at k:
                    #   If k < i, job moves left.
                    #   If k >= i, job moves right (or stays if k==i? No, if k==i, it goes to pos i in new list, which is old pos i+1? 
                    #   Let's trace: [A, B, C]. Remove B (i=1). Rest [A, C]. 
                    #   Insert B at k=0: [B, A, C] (Move left)
                    #   Insert B at k=1: [A, B, C] (Same)
                    #   Insert B at k=2: [A, C, B] (Move right)
                    # So skip k==i (which corresponds to original position)
                    if k == i:
                        continue
                        
                    new_perm = rest[:k] + [job] + rest[k:]
                    new_ms = _makespan_fast_optimized(pt_list, new_perm)
                    if new_ms < current_ms:
                        current = new_perm
                        current_ms = new_ms
                        improved = True
                        break # Restart scan from beginning
                if improved:
                    break
            
            if improved:
                continue # Go back to while loop, try again
                
            # --- Neighborhood 2: Reversal (2-Opt) ---
            # Reverse segment [i, j]
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    new_perm = current[:i] + current[i:j+1][::-1] + current[j+1:]
                    new_ms = _makespan_fast_optimized(pt_list, new_perm)
                    if new_ms < current_ms:
                        current = new_perm
                        current_ms = new_ms
                        improved = True
                        break # Restart scan from beginning
                if improved:
                    break
            
        return current, current_ms
    
    # Evaluate each candidate
    for cand in candidates:
        if time.time() > deadline:
            break
            
        perm, ms = _local_search(cand)
        if ms < best_makespan:
            best_makespan = ms
            best_perm = perm
    
    # If no candidate was processed (shouldn't happen), return identity
    if best_perm is None:
        best_perm = list(range(n_jobs))
        
    return best_perm

# EVOLVE-BLOCK-END
