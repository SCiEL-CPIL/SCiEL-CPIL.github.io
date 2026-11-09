"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, p):
    """Compute makespan for a given permutation. p shape: (n_jobs, n_machines)"""
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return 0.0
    machine_times = np.zeros(n_machines)
    total = 0.0
    for j in perm:
        prev = 0.0
        for m in range(n_machines):
            if machine_times[m] < prev:
                machine_times[m] = prev
            machine_times[m] += p[j, m]
            prev = machine_times[m]
        total = prev
    return total


def _makespan_fast(perm, p, machine_times=None):
    """Compute makespan with optional pre-allocated machine_times array. Returns (makespan, machine_times)."""
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return 0.0, machine_times if machine_times is not None else np.zeros(n_machines)
    if machine_times is None:
        machine_times = np.zeros(n_machines)
    else:
        machine_times[:] = 0.0
    total = 0.0
    for j in perm:
        prev = 0.0
        for m in range(n_machines):
            if machine_times[m] < prev:
                machine_times[m] = prev
            machine_times[m] += p[j, m]
            prev = machine_times[m]
        total = prev
    return total, machine_times


def _neh_order(p):
    """NEH heuristic: sort by total processing time descending, then insert each job at best position."""
    n_jobs, n_machines = p.shape
    totals = p.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build sequence incrementally
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _local_search(perm, p, max_iters=200, time_limit=10.0):
    """Adjacent swap local search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, p)
    start = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() - start > time_limit:
            break
        for i in range(n_jobs - 1):
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan(new_perm, p)
            if ms < best_ms:
                best_perm = new_perm
                best_ms = ms
                improved = True
                break
    return best_perm


def _vns_search(perm, p, time_limit=30.0):
    """Variable Neighborhood Search cycling through neighborhoods of increasing size."""
    n_jobs = len(perm)
    if n_jobs <= 2:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, p)
    start = time.time()
    
    # Neighborhood definitions:
    # N1: adjacent swaps
    # N2: 2-opt block reversals (reverse a segment of length 2-4)
    # N3: 3-opt segment relocations (move a segment of length 1-3 to a new position)
    
    neighborhood_sizes = [1, 2, 3]  # max segment length for each neighborhood
    
    while time.time() - start < time_limit:
        improved = False
        
        for k, max_seg in enumerate(neighborhood_sizes):
            if time.time() - start > time_limit:
                break
            
            if k == 0:
                # N1: Adjacent swaps
                for i in range(n_jobs - 1):
                    if time.time() - start > time_limit:
                        break
                    new_perm = list(best_perm)
                    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
                    ms = _makespan(new_perm, p)
                    if ms < best_ms:
                        best_perm = new_perm
                        best_ms = ms
                        improved = True
                        break
            elif k == 1:
                # N2: 2-opt block reversals (reverse segments of length 2-4)
                found_improvement = False
                for seg_len in range(2, min(max_seg + 1, n_jobs + 1)):
                    if time.time() - start > time_limit:
                        break
                    for i in range(n_jobs - seg_len + 1):
                        if time.time() - start > time_limit:
                            break
                        new_perm = list(best_perm)
                        # Reverse segment from i to i+seg_len-1
                        new_perm[i:i+seg_len] = new_perm[i:i+seg_len][::-1]
                        ms = _makespan(new_perm, p)
                        if ms < best_ms:
                            best_perm = new_perm
                            best_ms = ms
                            found_improvement = True
                            improved = True
                            break
                    if found_improvement:
                        break
            else:
                # N3: 3-opt segment relocations (move segment of length 1-3)
                found_improvement = False
                for seg_len in range(1, min(max_seg + 1, n_jobs + 1)):
                    if time.time() - start > time_limit:
                        break
                    for i in range(n_jobs - seg_len + 1):
                        if time.time() - start > time_limit:
                            break
                        segment = best_perm[i:i+seg_len]
                        # Remove segment
                        remaining = best_perm[:i] + best_perm[i+seg_len:]
                        # Try inserting at all positions
                        for j in range(len(remaining) + 1):
                            if time.time() - start > time_limit:
                                break
                            if j == i or (j > i and j <= i + seg_len):
                                continue  # Skip same position or overlapping
                            new_perm = remaining[:j] + segment + remaining[j:]
                            ms = _makespan(new_perm, p)
                            if ms < best_ms:
                                best_perm = new_perm
                                best_ms = ms
                                found_improvement = True
                                improved = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
        
        if not improved:
            # No improvement in any neighborhood, break
            break
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Parameters:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    
    Returns:
        list of job indices (permutation)
    """
    p = np.asarray(processing_times)
    n_jobs, n_machines = p.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    global_start = time.time()
    total_time_limit = 145.0  # Safety margin under 150s wall-clock
    
    # Phase 1: NEH heuristic (fast, good starting point)
    neh_perm = _neh_order(p)
    best_perm = list(neh_perm)
    best_ms = _makespan(best_perm, p)
    
    # Phase 2: Local search from NEH
    elapsed = time.time() - global_start
    remaining = total_time_limit - elapsed
    ls_time = min(remaining * 0.1, 5.0)
    if ls_time > 0.5:
        ls_perm = _local_search(best_perm, p, max_iters=300, time_limit=ls_time)
        ls_ms = _makespan(ls_perm, p)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Phase 3: VNS search from best solution
    elapsed = time.time() - global_start
    remaining = total_time_limit - elapsed
    if remaining > 5.0:
        vns_time = min(remaining * 0.5, 40.0)
        if vns_time > 1.0:
            vns_perm = _vns_search(best_perm, p, time_limit=vns_time)
            vns_ms = _makespan(vns_perm, p)
            if vns_ms < best_ms:
                best_perm = vns_perm
                best_ms = vns_ms
    
    # Phase 4: Second VNS from NEH with different starting point
    elapsed = time.time() - global_start
    remaining = total_time_limit - elapsed
    if remaining > 5.0:
        vns_time = min(remaining * 0.4, 30.0)
        if vns_time > 1.0:
            vns_perm2 = _vns_search(neh_perm, p, time_limit=vns_time)
            vns_ms2 = _makespan(vns_perm2, p)
            if vns_ms2 < best_ms:
                best_perm = vns_perm2
                best_ms = vns_ms2
    
    # Phase 5: Final local search polish
    elapsed = time.time() - global_start
    remaining = total_time_limit - elapsed
    if remaining > 3.0:
        ls_time = min(remaining * 0.3, 10.0)
        if ls_time > 0.5:
            ls_perm2 = _local_search(best_perm, p, max_iters=200, time_limit=ls_time)
            ls_ms2 = _makespan(ls_perm2, p)
            if ls_ms2 < best_ms:
                best_perm = ls_perm2
                best_ms = ls_ms2
    
    return list(best_perm)
# EVOLVE-BLOCK-END
