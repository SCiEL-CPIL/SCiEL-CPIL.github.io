"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    P = processing_times[perm]
    C = np.zeros((n_jobs, n_machines))
    C[:, 0] = np.cumsum(P[:, 0])
    for m in range(1, n_machines):
        for j in range(n_jobs):
            C[j, m] = max(C[j-1, m] if j > 0 else 0, C[j, m-1]) + P[j, m]
    return float(C[-1, -1])


def _compute_prefix_completions(perm, processing_times):
    """
    Compute completion time vectors for every prefix of the permutation.
    prefix_comps[i] is the completion vector after processing perm[0..i].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps
    
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps


def _compute_suffix_completions(perm, processing_times):
    """
    Compute "backward" completion vectors for every suffix of the permutation.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return [np.zeros(n_machines)]
    
    suffix_comps = [None] * (n_jobs + 1)
    suffix_comps[n_jobs] = np.zeros(n_machines)
    
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        
        W = np.zeros(n_machines)
        for m in range(n_machines - 1, -1, -1):
            if m == n_machines - 1:
                W[m] = pt[m]
            else:
                W[m] = pt[m] + max(W[m+1], 0)
        
        suffix_comps[i] = W + np.max(suffix_comps[i+1])
        
    return suffix_comps


def _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of swapping perm[i] and perm[i+1].
    """
    n_machines = processing_times.shape[1]
    
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1]
    
    job_a = perm[i]
    job_b = perm[i+1]
    
    pt_b = processing_times[job_b]
    pt_a = processing_times[job_a]
    
    C_b = np.zeros(n_machines)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = max(C_prev[m], C_b[m-1]) + pt_b[m]
        
    C_a = np.zeros(n_machines)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = max(C_b[m], C_a[m-1]) + pt_a[m]
        
    if i + 2 >= len(perm):
        return float(np.max(C_a))
    else:
        D_suffix = suffix_comps[i+2]
        return float(np.max(C_a + D_suffix))


def _eval_insert_incremental(perm, i, j, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of moving perm[i] to position j.
    """
    n_machines = processing_times.shape[1]
    
    if i == j:
        return _makespan(perm, processing_times)
    
    if i < j:
        if i == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[i-1]
            
        C_mid = C_prev
        for k in range(i+1, j+1):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_mid[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_mid[m], C_a[m-1]) + pt_a[m]
            
        if j + 1 >= len(perm):
            return float(np.max(C_a))
        else:
            D_suffix = suffix_comps[j+1]
            return float(np.max(C_a + D_suffix))
            
    else:
        if j == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[j-1]
            
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_prev[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_prev[m], C_a[m-1]) + pt_a[m]
            
        C_mid = C_a
        for k in range(j, i):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        if i + 1 >= len(perm):
            return float(np.max(C_mid))
        else:
            D_suffix = suffix_comps[i+1]
            return float(np.max(C_mid + D_suffix))


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    order = list(np.argsort(-totals))
    
    if n_jobs > 2:
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _precompute_swap_deltas(perm, processing_times):
    """
    Precompute all adjacent swap deltas using prefix/suffix completions.
    Returns array of delta values for each adjacent swap position.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return np.array([])
    
    prefix_comps = _compute_prefix_completions(perm, processing_times)
    suffix_comps = _compute_suffix_completions(perm, processing_times)
    
    current_ms = _makespan(perm, processing_times)
    deltas = np.zeros(n_jobs - 1)
    
    for i in range(n_jobs - 1):
        ms_after = _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps)
        deltas[i] = ms_after - current_ms
    
    return deltas


def _fast_local_search(perm, processing_times, max_iterations=500, deadline=None):
    """
    Fast local search using precomputed swap deltas for O(1) adjacent swap evaluation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Precompute all adjacent swap deltas
        deltas = _precompute_swap_deltas(best_perm, processing_times)
        
        improved = False
        
        # Find best adjacent swap (most negative delta)
        min_idx = np.argmin(deltas)
        if deltas[min_idx] < -1e-10:
            best_perm[min_idx], best_perm[min_idx+1] = best_perm[min_idx+1], best_perm[min_idx]
            best_ms += deltas[min_idx]
            improved = True
        
        if not improved:
            # Try insert moves
            prefix_comps = _compute_prefix_completions(best_perm, processing_times)
            suffix_comps = _compute_suffix_completions(best_perm, processing_times)
            
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = best_perm[i]
                for pos in range(n_jobs):
                    if pos == i or pos == i + 1:
                        continue
                    ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                    if ms < best_ms - 1e-10:
                        new_perm = _move_job(best_perm, job, pos)
                        best_perm = new_perm
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    return best_perm


def _high_precision_polish(perm, processing_times, deadline=None):
    """
    High-precision polish pass with exact local search and tighter convergence.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    # Phase 1: Exhaustive 2-opt (adjacent swaps) until convergence
    for it in range(1000):
        if deadline is not None and time.time() > deadline:
            break
        
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)
        
        improved = False
        for i in range(n_jobs - 1):
            ms_after = _eval_swap_incremental(best_perm, i, processing_times, prefix_comps, suffix_comps)
            if ms_after < best_ms - 1e-12:
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms = ms_after
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 2: Insert moves with exact evaluation
    for it in range(500):
        if deadline is not None and time.time() > deadline:
            break
        
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)
        
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                if ms < best_ms - 1e-12:
                    new_perm = _move_job(best_perm, job, pos)
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Phase 3: Final adjacent swap pass with very tight tolerance
    for it in range(200):
        if deadline is not None and time.time() > deadline:
            break
        
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)
        
        improved = False
        for i in range(n_jobs - 1):
            ms_after = _eval_swap_incremental(best_perm, i, processing_times, prefix_comps, suffix_comps)
            if ms_after < best_ms - 1e-14:
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms = ms_after
                improved = True
                break
        
        if not improved:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with fast incremental makespan updates for adjacent swaps.
    Uses prefix/suffix completions for O(n_machines) swap evaluation.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Adaptive temperature based on problem scale
    initial_temp = current_ms * 0.08
    min_temp = 1e-6
    cooling_rate = 0.999
    restart_count = 0
    max_restarts = 8
    
    temp = initial_temp
    steps_since_improve = 0
    improve_threshold = max(30, n_jobs)
    
    # Precompute prefix/suffix for current state
    prefix_comps = _compute_prefix_completions(current_perm, processing_times)
    suffix_comps = _compute_suffix_completions(current_perm, processing_times)
    
    while time.time() < deadline:
        if temp < min_temp:
            if restart_count >= max_restarts:
                break
            # Restart with higher temperature
            temp = current_ms * 0.06
            restart_count += 1
            steps_since_improve = 0
            # Perturb current solution
            for _ in range(min(3, n_jobs // 3)):
                i = rng.randint(0, n_jobs - 2)
                current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            current_ms = _makespan(current_perm, processing_times)
            prefix_comps = _compute_prefix_completions(current_perm, processing_times)
            suffix_comps = _compute_suffix_completions(current_perm, processing_times)
            continue
        
        # Choose move type: 60% adjacent swap, 40% insert
        move_type = rng.random()
        if move_type < 0.6:
            i = rng.randint(0, n_jobs - 2)
            # Fast incremental evaluation for adjacent swap
            new_ms = _eval_swap_incremental(current_perm, i, processing_times, prefix_comps, suffix_comps)
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
                current_ms = new_ms
                # Update prefix/suffix only if accepted
                prefix_comps = _compute_prefix_completions(current_perm, processing_times)
                suffix_comps = _compute_suffix_completions(current_perm, processing_times)
                if current_ms < best_ms - 1e-10:
                    best_perm = list(current_perm)
                    best_ms = current_ms
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                steps_since_improve += 1
        else:
            # Insert move: pick random job and random position
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                steps_since_improve += 1
                continue
            job = current_perm[i]
            
            # Use incremental evaluation for insert
            new_ms = _eval_insert_incremental(current_perm, i, j, processing_times, prefix_comps, suffix_comps)
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                current_perm = _move_job(current_perm, job, j)
                current_ms = new_ms
                # Update prefix/suffix
                prefix_comps = _compute_prefix_completions(current_perm, processing_times)
                suffix_comps = _compute_suffix_completions(current_perm, processing_times)
                if current_ms < best_ms - 1e-10:
                    best_perm = list(current_perm)
                    best_ms = current_ms
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        # Cool down
        temp *= cooling_rate
        
        # If stuck too long, restart with perturbation
        if steps_since_improve > improve_threshold * 2:
            # Perturb current solution
            for _ in range(min(4, n_jobs // 2)):
                i = rng.randint(0, n_jobs - 2)
                current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            current_ms = _makespan(current_perm, processing_times)
            temp = current_ms * 0.05
            steps_since_improve = 0
            restart_count += 1
            prefix_comps = _compute_prefix_completions(current_perm, processing_times)
            suffix_comps = _compute_suffix_completions(current_perm, processing_times)
            if restart_count > max_restarts:
                break
    
    return best_perm


def _tabu_search(perm, processing_times, deadline, rng=None):
    """
    Tabu search with insert and swap moves, using incremental evaluation.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Tabu list: store (job, position) pairs with expiration
    tabu_list = {}
    tabu_tenure = max(10, n_jobs // 2)
    
    # Precompute prefix/suffix
    prefix_comps = _compute_prefix_completions(current_perm, processing_times)
    suffix_comps = _compute_suffix_completions(current_perm, processing_times)
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = max(100, n_jobs * 5)
    
    while time.time() < deadline:
        iteration += 1
        
        # Update tabu list expiration
        if iteration % 10 == 0:
            keys_to_remove = []
            for key, exp in tabu_list.items():
                if iteration > exp:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del tabu_list[key]
        
        # Find best non-tabu move
        best_move = None
        best_move_ms = float('inf')
        
        # Try insert moves
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_perm[i]
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                # Check tabu
                if (job, pos) in tabu_list:
                    # Aspiration criterion: allow if it improves best
                    ms = _eval_insert_incremental(current_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                    if ms < best_ms - 1e-10:
                        # Allow this move despite tabu
                        pass
                    else:
                        continue
                else:
                    ms = _eval_insert_incremental(current_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = ('insert', i, pos)
        
        # Try adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            ms = _eval_swap_incremental(current_perm, i, processing_times, prefix_comps, suffix_comps)
            if ms < best_move_ms:
                best_move_ms = ms
                best_move = ('swap', i, i + 1)
        
        if best_move is None:
            break
        
        # Apply best move
        if best_move[0] == 'insert':
            _, i, pos = best_move
            job = current_perm[i]
            current_perm = _move_job(current_perm, job, pos)
            # Add to tabu
            tabu_list[(job, pos)] = iteration + tabu_tenure
        else:
            _, i, j = best_move
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            # Add to tabu
            tabu_list[(current_perm[i], i)] = iteration + tabu_tenure
            tabu_list[(current_perm[j], j)] = iteration + tabu_tenure
        
        current_ms = best_move_ms
        prefix_comps = _compute_prefix_completions(current_perm, processing_times)
        suffix_comps = _compute_suffix_completions(current_perm, processing_times)
        
        if current_ms < best_ms - 1e-10:
            best_perm = list(current_perm)
            best_ms = current_ms
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Diversification: if stuck for too long, perturb
        if no_improve_count > max_no_improve:
            for _ in range(min(5, n_jobs // 2)):
                i = rng.randint(0, n_jobs - 2)
                current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            current_ms = _makespan(current_perm, processing_times)
            prefix_comps = _compute_prefix_completions(current_perm, processing_times)
            suffix_comps = _compute_suffix_completions(current_perm, processing_times)
            no_improve_count = 0
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    candidates = []
    
    # LPT (Longest Processing Time) - sort by total processing time descending
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # SPT (Shortest Processing Time) - sort by total processing time ascending
    spt_perm = list(np.argsort(processing_times.sum(axis=1)))
    candidates.append(spt_perm)
    
    # NEH heuristic
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    # Second NEH with different seed
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    # NEH with forward-backward
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Random permutation for diversity
    if time.time() < deadline:
        rand_perm = list(range(n_jobs))
        rng.shuffle(rand_perm)
        candidates.append(rand_perm)
    
    # Evaluate all candidates
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Local search phase
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        if n_jobs <= 30:
            max_iter = 2000
        elif n_jobs <= 50:
            max_iter = 1000
        elif n_jobs <= 100:
            max_iter = 500
        else:
            max_iter = 200
        
        polished = _fast_local_search(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Simulated annealing phase - use remaining time wisely
    if time.time() < deadline - 15.0 and n_jobs > 1:
        sa_deadline = deadline - 15.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Final local search after SA
        if time.time() < deadline - 10.0:
            max_iter_final = 500 if n_jobs <= 50 else 200
            final_polish = _fast_local_search(best_perm, processing_times, max_iterations=max_iter_final, deadline=deadline - 10.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    # Tabu search phase
    if time.time() < deadline - 10.0 and n_jobs > 1:
        ts_deadline = deadline - 10.0
        ts_perm = _tabu_search(best_perm, processing_times, ts_deadline, rng)
        ts_ms = _makespan(ts_perm, processing_times)
        if ts_ms < best_ms:
            best_perm = ts_perm
            best_ms = ts_ms
    
    # High-precision polish pass
    if time.time() < deadline - 3.0:
        hp_deadline = deadline - 3.0
        hp_polish = _high_precision_polish(best_perm, processing_times, deadline=hp_deadline)
        hp_ms = _makespan(hp_polish, processing_times)
        if hp_ms < best_ms:
            best_perm = hp_polish
            best_ms = hp_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
