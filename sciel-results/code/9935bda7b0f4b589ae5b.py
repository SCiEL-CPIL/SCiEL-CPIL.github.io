"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """
    Compute makespan for a given job order.
    Optimized for speed in Python by using local variables and minimal overhead.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    
    completion = [0.0] * n_machines
    
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            # p[m] is accessed from numpy array, which is slightly slower than list
            # But converting pt to list of lists might be faster for small N?
            # Let's stick to numpy for now, or convert once in order_jobs.
            
            if prev < completion[m]:
                prev = completion[m]
            prev += p[m]
            completion[m] = prev
    return completion[-1]


def _neh_initialization(processing_times, seed=None):
    """
    NEH heuristic with randomization for diversity.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Order jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    
    # Add randomization to break ties and create diversity
    # The perturbation magnitude is crucial. Too small = same order, too large = random.
    # Scale perturbation by the range of totals.
    range_totals = totals.max() - totals.min()
    if range_totals < 1e-9:
        range_totals = 1.0
        
    # Use a small fraction of the range for perturbation
    perturbation = rng.rand(n_jobs) * 0.1 * range_totals
    perturbed_totals = totals + perturbation
    
    job_order = list(np.argsort(-perturbed_totals))
    
    # Build sequence greedily using NEH rule
    seq = [job_order[0]]
    
    for job in job_order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            # Evaluate insertion
            # To speed this up, we can use a helper that doesn't create new lists if possible,
            # but for N < 100, list creation is acceptable.
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _local_search(order, processing_times, time_limit=50.0):
    """
    Local search using insertion moves with First Improvement strategy.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(order, processing_times)
    
    best_order = list(order)
    best_ms = _makespan(best_order, processing_times)
    
    # Convert processing_times to a list of lists for faster access in the inner loop?
    # pt_list = processing_times.tolist()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Iterate through all positions
        for i in range(n_jobs):
            if time.time() - start_time >= time_limit:
                return best_order, best_ms
            
            job = best_order[i]
            # Remove job from current position
            # We don't actually remove it, we just consider inserting it elsewhere
            # But we must skip the original position and the position immediately after removal (which is i)
            
            # Try inserting at all other positions
            # If we insert at j, the new sequence is:
            # best_order[:i] + best_order[i+1:j+1] + [job] + best_order[j+1:] (if j > i)
            # or best_order[:j] + [job] + best_order[j:i] + best_order[i+1:] (if j < i)
            
            # To avoid complex slicing logic, we can construct the new list.
            # For efficiency, we can limit the search range if N is large?
            # For now, try all j.
            
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                
                # Construct new order
                # This is O(N) per move.
                # Total O(N^3) per pass.
                
                if j < i:
                    # Move job from i to j
                    # Elements j..i-1 shift right
                    new_order = best_order[:j] + [job] + best_order[j:i] + best_order[i+1:]
                else:
                    # j > i + 1
                    # Elements i+1..j shift left
                    new_order = best_order[:i] + best_order[i+1:j+1] + [job] + best_order[j+1:]
                
                ms = _makespan(new_order, processing_times)
                
                if ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    # First Improvement: restart the loop or continue?
                    # Restarting is safer to ensure we don't miss better moves in the new configuration.
                    # But it might be slow.
                    # Let's break the inner loop and restart the outer loop.
                    break
            
            # If we found an improvement and broke the inner loop, 
            # we should restart the outer loop to evaluate the new sequence from the beginning.
            if improved:
                # We already set improved=True. 
                # To restart, we can use a flag or just break out of the for loop and let the while loop continue.
                # But we need to reset the 'improved' flag for the next while iteration?
                # No, 'improved' is the flag for the while loop. 
                # If we break the inner loop, we exit the for loop.
                # The while loop condition checks 'improved'. It is True, so it loops again.
                # This effectively restarts the pass.
                pass

    return best_order, best_ms


def _block_decomposition(processing_times, n_blocks=None):
    """
    Partition jobs into blocks based on processing time similarity.
    Returns list of blocks, each a list of job indices.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    
    if n_blocks is None:
        n_blocks = max(2, int(np.sqrt(n_jobs)))
    
    # Compute job profiles
    totals = processing_times.sum(axis=1)
    means = processing_times.mean(axis=1)
    
    # Sort by total processing time
    sorted_jobs = np.argsort(totals)
    
    # Divide into roughly equal-sized blocks
    block_size = n_jobs // n_blocks
    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n_jobs
        blocks.append(sorted_jobs[start:end].tolist())
    
    return blocks


def _optimize_blocks(blocks, processing_times, time_limit=50.0):
    """
    Optimize internal order of each block using NEH, then optimize block order.
    """
    start_time = time.time()
    
    # Step 1: Optimize internal order of each block
    optimized_blocks = []
    for block in blocks:
        if len(block) <= 1:
            optimized_blocks.append(block)
            continue
        
        # Create sub-matrix for this block
        block_jobs = np.array(block)
        sub_pt = processing_times[block_jobs]
        
        # Use NEH on the sub-problem
        sub_order = _neh_initialization(sub_pt)
        # Map back to original job indices
        optimized_blocks.append([block_jobs[j] for j in sub_order])
    
    # Step 2: Determine optimal block order using 2-opt
    n_blocks = len(optimized_blocks)
    if n_blocks <= 1:
        return optimized_blocks[0] if n_blocks == 1 else []
    
    # Compute block makespan contribution
    block_totals = np.array([
        sum(processing_times[j] for j in block) for block in optimized_blocks
    ])
    
    # Start with blocks in current order
    block_order = list(range(n_blocks))
    
    # 2-opt on block order
    def _block_makespan(order_of_blocks):
        full_order = []
        for b_idx in order_of_blocks:
            full_order.extend(optimized_blocks[b_idx])
        return _makespan(full_order, processing_times)
    
    best_block_order = list(block_order)
    best_ms = _block_makespan(best_block_order)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n_blocks - 1):
            for j in range(i + 1, n_blocks):
                if time.time() - start_time >= time_limit:
                    break
                
                # Reverse blocks i..j
                new_order = best_block_order[:]
                new_order[i:j+1] = reversed(new_order[i:j+1])
                
                ms = _block_makespan(new_order)
                if ms < best_ms - 1e-9:
                    best_block_order = new_order
                    best_ms = ms
                    improved = True
        
        if not improved:
            break
    
    # Reconstruct final order
    final_order = []
    for b_idx in best_block_order:
        final_order.extend(optimized_blocks[b_idx])
    
    return final_order


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses multiple NEH restarts, block decomposition, and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (5s before 120s limit)
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    best_order = None
    best_ms = float('inf')
    
    # Strategy 1: Multiple NEH restarts with local search
    # Increase restarts if N is small, as local search is faster
    n_restarts = 20 if n_jobs < 30 else 10
    time_per_restart = (deadline - start_time - 10.0) / max(1, n_restarts)
    
    for r in range(n_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        # Generate diverse initial solution
        seed = r * 12345 + 99
        initial = _neh_initialization(processing_times, seed=seed)
        
        # Apply local search
        remaining = min(time_per_restart, deadline - time.time() - 10.0)
        if remaining > 1.0:
            improved, ms = _local_search(initial, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
    
    # Strategy 2: Block-based decomposition (if time remains)
    if time.time() < deadline - 5.0 and n_jobs >= 4:
        remaining = deadline - time.time() - 5.0
        blocks = _block_decomposition(processing_times)
        block_solution = _optimize_blocks(blocks, processing_times, time_limit=remaining)
        
        if block_solution and len(block_solution) == n_jobs:
            ms = _makespan(block_solution, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_order = block_solution
    
    # Final polish on best solution found
    if best_order is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 2.0:
            polished, ms = _local_search(best_order, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = polished
    
    if best_order is None:
        # Fallback: simple NEH
        best_order = _neh_initialization(processing_times)
    
    return best_order
# EVOLVE-BLOCK-END
