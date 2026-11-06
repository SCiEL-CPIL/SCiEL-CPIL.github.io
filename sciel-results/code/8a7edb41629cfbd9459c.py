"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _compute_completion_times(perm, processing_times):
    """Compute full completion time matrix for a permutation.
    Returns array of shape (n_jobs, n_machines) where C[j][m] is completion time of job j on machine m.
    Uses vectorized numpy where possible.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return np.zeros((0, n_machines), dtype=np.float64)
    
    pt_matrix = processing_times[np.array(perm)]  # shape (n_jobs, n_machines)
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    # Machine 0: cumulative sum
    C[:, 0] = np.cumsum(pt_matrix[:, 0])
    
    # Remaining machines: C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    # This has a dependency chain along both j and m, so we iterate over jobs
    # but vectorize over machines within each job
    for j in range(1, n_jobs):
        # C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m] for m=1..n_machines-1
        # C[j][m-1] is already computed for this j
        C[j, 1:] = np.maximum(C[j-1, 1:], C[j, :-1]) + pt_matrix[j, 1:]
    
    return C


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    C = _compute_completion_times(perm, processing_times)
    return float(C[-1, -1])


def _makespan_after_move(perm, C, i, j, processing_times):
    """Compute new makespan after moving job from position i to position j.
    Only recomputes affected portion. i and j are positions in the permutation.
    Uses vectorized numpy for speed.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return float(C[-1, -1])
    
    # Create new permutation
    job = perm[i]
    if j > i:
        new_perm = list(perm[:i]) + list(perm[i+1:j+1]) + [job] + list(perm[j+1:])
    else:
        new_perm = list(perm[:j]) + [job] + list(perm[j:i]) + list(perm[i+1:])
    
    start_pos = min(i, j)
    
    # Recompute completion times from start_pos onwards
    C_new = C.copy()
    pt_matrix = processing_times[np.array(new_perm)]
    
    for pos in range(start_pos, n_jobs):
        if pos == 0:
            C_new[pos, 0] = pt_matrix[pos, 0]
            if n_machines > 1:
                C_new[pos, 1:] = np.cumsum(pt_matrix[pos, 1:])
        else:
            C_new[pos, 0] = C_new[pos-1, 0] + pt_matrix[pos, 0]
            if n_machines > 1:
                C_new[pos, 1:] = np.maximum(C_new[pos-1, 1:], C_new[pos, :-1]) + pt_matrix[pos, 1:]
    
    return float(C_new[-1, -1])


def _neh_order(processing_times):
    """Generate NEH heuristic ordering."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward NEH: insert jobs one by one at best position
    partial = [order[0]]
    C_partial = _compute_completion_times(partial, processing_times)
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
        C_partial = _compute_completion_times(partial, processing_times)
    
    return partial


def _neh_order_perturbed(processing_times, rng, perturbation_strength=0.3):
    """Generate NEH ordering with perturbed job order.
    Uses a mix of swaps and insertions for more diverse perturbations.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    base_order = list(np.argsort(-totals))
    perturbed_order = base_order.copy()
    
    # Apply random swaps and random re-insertions for more diversity
    n_swaps = max(1, int(n_jobs * perturbation_strength * 0.5))
    n_inserts = max(1, int(n_jobs * perturbation_strength * 0.5))
    
    for _ in range(n_swaps):
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
    
    for _ in range(n_inserts):
        i = rng.randint(0, n_jobs - 1)
        job = perturbed_order.pop(i)
        j = rng.randint(0, n_jobs - 1)
        perturbed_order.insert(j, job)
    
    # Forward NEH: insert jobs one by one at best position
    partial = [perturbed_order[0]]
    for i in range(1, n_jobs):
        job = perturbed_order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    
    return partial


def _insertion_local_search(perm, processing_times, deadline, max_passes=100):
    """Local search using insertion moves with vectorized incremental makespan."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm), _makespan(perm, processing_times)
    
    best_perm = list(perm)
    C = _compute_completion_times(best_perm, processing_times)
    best_ms = float(C[-1, -1])
    
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                
                ms = _makespan_after_move(best_perm, C, i, j, processing_times)
                if ms < best_ms - 1e-10:
                    if j > i:
                        best_perm = best_perm[:i] + best_perm[i+1:j+1] + [job] + best_perm[j+1:]
                    else:
                        best_perm = best_perm[:j] + [job] + best_perm[j:i] + best_perm[i+1:]
                    best_ms = ms
                    C = _compute_completion_times(best_perm, processing_times)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_perm, best_ms


def _swap_local_search(perm, processing_times, deadline, max_passes=50):
    """Local search using adjacent swap moves with vectorized makespan."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm), _makespan(perm, processing_times)
    
    best_perm = list(perm)
    C = _compute_completion_times(best_perm, processing_times)
    best_ms = float(C[-1, -1])
    
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms - 1e-10:
                best_perm = new_perm
                best_ms = ms
                C = _compute_completion_times(best_perm, processing_times)
                improved = True
        
        if not improved:
            break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 5s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    
    rng = random.Random(42)
    
    # Baseline: longest processing time first
    baseline_perm = list(np.argsort(-processing_times.sum(axis=1)))
    best_perm = baseline_perm
    best_ms = _makespan(best_perm, processing_times)
    
    # Try NEH heuristic
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times)
        neh_ms = _makespan(neh_perm, processing_times)
        if neh_ms < best_ms:
            best_perm = neh_perm
            best_ms = neh_ms
    
    # Multi-start: try perturbed NEH orders with varying perturbation strengths
    if time.time() < deadline:
        n_starts = max(3, min(20, int((deadline - time.time()) / 0.3)))
        strengths = [0.15, 0.25, 0.35, 0.5, 0.65, 0.8]
        for k in range(n_starts):
            if time.time() >= deadline:
                break
            
            strength = strengths[k % len(strengths)]
            perturbed = _neh_order_perturbed(processing_times, rng, perturbation_strength=strength)
            p_ms = _makespan(perturbed, processing_times)
            if p_ms < best_ms:
                best_perm = perturbed
                best_ms = p_ms
    
    # Local search: insertion moves first (more powerful), then swaps
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        
        # Phase 1: Insertion local search
        if remaining > 5.0:
            insert_deadline = start_time + remaining * 0.7
            polished, polished_ms = _insertion_local_search(
                best_perm, processing_times, insert_deadline, max_passes=200
            )
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
        
        # Phase 2: Swap local search for fine-tuning
        if time.time() < deadline:
            swap_deadline = start_time + (deadline - time.time()) * 0.5
            polished, polished_ms = _swap_local_search(
                best_perm, processing_times, swap_deadline, max_passes=500
            )
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
        
        # Phase 3: Another round of insertion to refine further
        if time.time() < deadline:
            polished, polished_ms = _insertion_local_search(
                best_perm, processing_times, deadline, max_passes=100
            )
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
