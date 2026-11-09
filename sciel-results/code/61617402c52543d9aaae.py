"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a permutation in a flow-shop."""
    n_jobs, n_machines = pt.shape
    # C[i] = completion time of job perm[i] on each machine
    # Standard flow-shop: C[i][m] = max(C[i-1][m], C[i][m-1]) + pt[perm[i], m]
    # We can track just the last row and previous row, or use 2D for clarity
    # For efficiency, track C_prev_machine[job_index_in_perm] and C_current
    # Actually, simplest correct approach:
    # C[i][m] for i-th job in perm, m-th machine
    # C[0][0] = pt[perm[0], 0]
    # C[i][0] = C[i-1][0] + pt[perm[i], 0]  (first machine, sequential)
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + pt[perm[i], m]
    
    # Use 1D arrays for speed: prev_mach[job_pos] = C[job_pos][m-1]
    # cur_mach[job_pos] = C[job_pos][m]
    # But we need C[job_pos-1][m] which is cur_mach[job_pos-1]
    
    n = len(perm)
    if n == 0:
        return 0.0
    
    # prev_mach[i] = completion time of job at position i on machine m-1
    # cur_mach[i] = completion time of job at position i on machine m
    prev_mach = np.zeros(n)
    cur_mach = np.zeros(n)
    
    for m in range(n_machines):
        for i in range(n):
            if i == 0:
                if m == 0:
                    cur_mach[0] = pt[perm[0], m]
                else:
                    cur_mach[0] = prev_mach[0] + pt[perm[0], m]
            else:
                if m == 0:
                    cur_mach[i] = cur_mach[i-1] + pt[perm[i], m]
                else:
                    cur_mach[i] = max(prev_mach[i], cur_mach[i-1]) + pt[perm[i], m]
        # After processing machine m, cur_mach becomes prev_mach for next machine
        prev_mach, cur_mach = cur_mach, prev_mach
        # Reset cur_mach (it's now the old prev_mach, will be overwritten)
    
    # The makespan is the completion time of the last job on the last machine
    # After the loop, if n_machines is even, the result is in cur_mach (which was swapped)
    # Let me trace: after machine m, we swap so prev_mach holds C[.][m]
    # So after last machine (m = n_machines-1), prev_mach holds C[.][n_machines-1]
    # Makespan = prev_mach[-1]
    
    return float(prev_mach[-1])


def _neh_ordering(pt):
    """NEH heuristic for flow-shop scheduling."""
    n_jobs, n_machines = pt.shape
    total_pt = pt.sum(axis=1)
    
    # Sort jobs by total processing time in descending order
    order = np.argsort(-total_pt)
    
    # Insert each job into the best position
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return perm


def _insertion_local_search(perm, pt, max_iterations=None):
    """Insertion-based local search: try moving each job to every other position."""
    n = len(perm)
    if n == 0:
        return perm, float('inf')
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, pt)
    
    if max_iterations is None:
        max_iterations = 1000
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(n):
            job = best_perm[i]
            for j in range(n):
                if i == j:
                    continue
                # Move job from position i to position j
                new_perm = best_perm[:i] + best_perm[i+1:j+1] + [job] + best_perm[j+1:]
                ms = _makespan(new_perm, pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    # After improvement, restart the inner loop from beginning
                    # for more effective search
                    break
            if improved:
                break
    
    return best_perm, best_ms


def _perturb(perm, rng, strength=3):
    """Perturb a permutation by swapping a few random pairs."""
    perm = list(perm)
    n = len(perm)
    if n < 2:
        return perm
    for _ in range(strength):
        i, j = rng.choice(n, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def order_jobs(processing_times):
    """Order jobs to minimize makespan on a flow-shop."""
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin from 160s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: NEH heuristic for initial solution
    best_perm = _neh_ordering(pt)
    best_ms = _makespan(best_perm, pt)
    
    # Phase 2: Local search from NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, pt, max_iterations=500)
    
    # Check time
    if time.time() > deadline:
        return list(best_perm)
    
    # Phase 3: Iterated Local Search (ILS) with multiple restarts
    perturbation_strength = 2
    
    while time.time() < deadline:
        # Perturb the best solution
        perturbed = _perturb(best_perm, rng, strength=perturbation_strength)
        
        # Local search from perturbed solution
        perm_ls, ms_ls = _insertion_local_search(perturbed, pt, max_iterations=200)
        
        if ms_ls < best_ms:
            best_ms = ms_ls
            best_perm = perm_ls
            # Reduce perturbation on improvement
            perturbation_strength = max(1, perturbation_strength - 1)
        else:
            # Accept with some probability (simulated annealing style)
            # Or just reject and increase perturbation
            perturbation_strength = min(5, perturbation_strength + 1)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
