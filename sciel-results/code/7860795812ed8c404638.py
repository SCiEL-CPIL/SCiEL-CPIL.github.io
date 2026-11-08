"""Flow-shop makespan minimizer: NEH constructive + CIG + local search with time guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan for a permutation. O(n*m) via cumulative sums on selected rows."""
    # perm is array of job indices, pt is (n_jobs, n_machines)
    selected = pt[perm]
    # Cumulative sum along jobs axis, then take the last element of each machine column
    # Actually makespan is the max completion time over all machines
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # The makespan is C[n-1][m-1]
    # Efficient computation:
    # For each machine m, the completion time of the last job is:
    # C_last[m] = max over j of (sum_{k<=j} p[k][m] + max over m' < m of (sum_{k<=j-1} p[k][m']))
    # Simpler: just do the DP
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
    
    # C[j] = completion time of job perm[j] on each machine
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[perm[j]][m]
    # Makespan = C[n_jobs-1][n_machines-1]
    
    # Use vectorized approach where possible
    # For small instances, direct computation is fine
    c = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        for m in range(n_machines):
            if m == 0:
                c[m] = c[m] + pt[job, m]
            else:
                c[m] = max(c[m], c[m-1]) + pt[job, m]
    return float(c[-1])


def _makespan_vectorized(perm, pt):
    """Faster makespan using vectorized operations where possible."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # For each machine, we need the cumulative sum of processing times in order
    # But the constraint is that machine m can only start job j after:
    #   1. The previous job on machine m is done
    #   2. The same job on machine m-1 is done
    # So C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    
    # Let's use a different approach: for each machine m, 
    # C[j][m] = max_{i<=j} (sum_{k=i}^{j} p[k][m] + C[i-1][m-1])
    # This is complex. Let's just do the O(n*m) DP but with numpy for the inner loop
    # Actually for typical instance sizes (n <= ~50, m <= ~10), the Python loop is fine
    
    selected = pt[perm]  # (n_jobs, n_machines)
    c = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        row = selected[j]
        for m in range(n_machines):
            if m == 0:
                c[m] += row[m]
            else:
                c[m] = max(c[m], c[m-1]) + row[m]
    return float(c[-1])


def _neh_constructive(pt):
    """NEH heuristic: sort by total processing time descending, insert each job in best position."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with empty sequence, insert jobs one by one
    seq = []
    for job in order:
        # Try all insertion positions
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_vectorized(np.array(candidate, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
    
    return np.array(seq, dtype=np.int64)


def _cig_constructive(pt):
    """CIG (Cambridge Insertion Heuristic) - similar to NEH but with different insertion criterion."""
    # CIG uses the same sorting but may differ in tie-breaking or insertion cost
    # For now, use NEH as base
    return _neh_constructive(pt)


def _local_search(perm, pt, time_budget=30.0):
    """Best-improvement local search with 2-opt, swap, and insert moves."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    best_perm = perm.copy()
    best_ms = _makespan_vectorized(best_perm, pt)
    
    start_time = time.time()
    improved = True
    
    while improved:
        improved = False
        current_time = time.time()
        if current_time - start_time > time_budget:
            break
        
        # Try all 2-opt moves (reverse segments)
        for i in range(n_jobs):
            for j in range(i + 2, n_jobs):
                candidate = best_perm.copy()
                candidate[i:j+1] = candidate[i:j+1][::-1]
                ms = _makespan_vectorized(candidate, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    if time.time() - start_time > time_budget:
                        return best_perm, best_ms
        
        # Try all swap moves
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                candidate = best_perm.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan_vectorized(candidate, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    if time.time() - start_time > time_budget:
                        return best_perm, best_ms
        
        # Try all insert moves (move job i to position j)
        for i in range(n_jobs):
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j or i == j + 1:
                    continue
                candidate = np.delete(best_perm, i)
                candidate = np.insert(candidate, j, job)
                ms = _makespan_vectorized(candidate, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    if time.time() - start_time > time_budget:
                        return best_perm, best_ms
    
    return best_perm, best_ms


def _random_permutation(n_jobs, rng):
    """Generate a random permutation."""
    return rng.permutation(n_jobs)


def _perturbation(perm, pt, rng, strength=3):
    """Apply random perturbation to escape local optima."""
    n_jobs = len(perm)
    perturbed = perm.copy()
    for _ in range(strength):
        i = rng.integers(0, n_jobs)
        j = rng.integers(0, n_jobs)
        # Insert perturbation: move job i to position j
        job = perturbed[i]
        perturbed = np.delete(perturbed, i)
        if j > i:
            j -= 1
        perturbed = np.insert(perturbed, j, job)
    return perturbed


def order_jobs(processing_times):
    """Main entry point: construct and optimize job permutation for flow-shop makespan."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Constructive heuristics
    candidates = []
    
    # NEH
    neh_perm = _neh_constructive(pt)
    neh_ms = _makespan_vectorized(neh_perm, pt)
    candidates.append((neh_perm, neh_ms))
    
    # Also try a few random starts
    n_random_starts = 5
    for _ in range(n_random_starts):
        if time.time() > deadline - 60:
            break
        rand_perm = _random_permutation(n_jobs, rng)
        rand_ms = _makespan_vectorized(rand_perm, pt)
        candidates.append((rand_perm, rand_ms))
    
    # Sort candidates by makespan
    candidates.sort(key=lambda x: x[1])
    
    best_perm = candidates[0][0]
    best_ms = candidates[0][1]
    
    # Phase 2: Local search from best candidate
    if time.time() < deadline - 30:
        ls_perm, ls_ms = _local_search(best_perm, pt, time_budget=min(30, deadline - time.time() - 5))
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Phase 3: Iterated local search for remaining time
    if time.time() < deadline - 10:
        il_perm = best_perm.copy()
        il_ms = best_ms
        no_improve_count = 0
        
        while time.time() < deadline - 5:
            # Perturb
            perturbed = _perturbation(il_perm, pt, rng, strength=2 + int(3 * (1 - (time.time() - start_time) / 145.0)))
            
            # Local search
            ls_perm, ls_ms = _local_search(perturbed, pt, time_budget=min(10, deadline - time.time() - 5))
            
            if ls_ms < il_ms - 1e-9:
                il_perm = ls_perm
                il_ms = ls_ms
                no_improve_count = 0
                if il_ms < best_ms:
                    best_perm = il_perm.copy()
                    best_ms = il_ms
            else:
                no_improve_count += 1
                # Accept worse solutions with some probability (simulated annealing-like)
                if no_improve_count < 20 and rng.random() < 0.3:
                    il_perm = ls_perm
                    il_ms = ls_ms
            
            if no_improve_count > 30:
                break
    
    return list(best_perm.astype(int))
# EVOLVE-BLOCK-END
