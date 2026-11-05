"""Flow-shop permutation optimizer using NEH + simulated annealing with or-opt and swap moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, p):
    """Compute makespan for a given permutation on processing times p."""
    n_jobs, n_machines = p.shape
    # cum[m] = completion time on machine m for the current job in sequence
    cum = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                cum[m] += p[j, m]
            else:
                cum[m] = max(cum[m - 1], cum[m]) + p[j, m]
    return cum[-1]


def _makespan_vec(perm, p):
    """Vectorized makespan computation for a single permutation."""
    n_jobs, n_machines = p.shape
    # Build the sequence matrix: shape (n_jobs, n_machines)
    seq = p[perm]  # (n_jobs, n_machines)
    # Cumulative along jobs, but each machine depends on previous machine
    # We can do this iteratively over machines
    cum = np.zeros(n_jobs)
    for m in range(n_machines):
        if m == 0:
            cum = np.cumsum(seq[:, m])
        else:
            # cum[m] = max(cum[m-1][i], cum[m-1][i-1]) + p[i,m]  (with cum[m-1][-1]=0 for first)
            prev = cum  # completion on machine m-1 for each job position
            # For position i, start time on machine m = max(completion of job i-1 on machine m, completion of job i on machine m-1)
            # completion of job i-1 on machine m is the previous value in the running sum
            # We need to compute: c[i] = max(c[i-1], prev[i]) + seq[i, m]
            # This is a sequential recurrence, can't fully vectorize without scan
            # Use a simple loop over jobs (n_jobs is small typically)
            new_cum = np.empty(n_jobs)
            c_prev = 0.0
            for i in range(n_jobs):
                start = max(c_prev, prev[i])
                c_prev = start + seq[i, m]
                new_cum[i] = c_prev
            cum = new_cum
    return cum[-1]


def _neh_initial(p):
    """NEH heuristic: insert jobs one by one at best position."""
    n_jobs, n_machines = p.shape
    # Sort by total processing time descending
    order = np.argsort(-p.sum(axis=1))
    perm = []
    for j in order:
        # Try inserting at each position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [j] + perm[best_pos:]
    return perm


def _neighborhood_moves(perm, p, rng):
    """Generate a random neighbor using swap or or-opt (segment move)."""
    n = len(perm)
    if n <= 1:
        return perm
    move_type = rng.random()
    if move_type < 0.5:
        # Swap two jobs
        i, j = rng.choice(n, size=2, replace=False)
        new_perm = perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm
    elif move_type < 0.8:
        # Or-opt: move a segment of length 1 or 2 to another position
        seg_len = 1 if rng.random() < 0.7 else 2
        if n < seg_len + 1:
            seg_len = 1
        i = rng.randint(0, n - seg_len + 1)
        # Choose destination (not adjacent to original)
        j = rng.randint(0, n - seg_len)
        if abs(i - j) <= seg_len:
            j = (j + n // 2) % max(1, n - seg_len)
        segment = perm[i:i + seg_len]
        rest = perm[:i] + perm[i + seg_len:]
        # Insert segment at position j in rest
        if j > i + seg_len:
            j -= seg_len
        new_perm = rest[:j] + list(segment) + rest[j:]
        return new_perm
    else:
        # 2-opt: reverse a segment
        i, j = rng.choice(n, size=2, replace=False)
        if i > j:
            i, j = j, i
        new_perm = perm[:i] + list(perm[i:j + 1][::-1]) + perm[j + 1:]
        return new_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using NEH + SA with or-opt/swap moves."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.default_rng(42)
    
    # Initial solution via NEH
    perm = _neh_initial(p)
    best_perm = perm.copy()
    best_ms = _makespan(perm, p)
    
    # Simulated annealing
    T0 = best_ms * 0.1 if best_ms > 0 else 1.0
    T = T0
    cooling = 0.9995
    T_min = 1e-6
    
    step = 0
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Generate neighbor
        new_perm = _neighborhood_moves(perm, p, rng)
        new_ms = _makespan(new_perm, p)
        
        delta = new_ms - best_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            perm = new_perm
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = new_perm.copy()
        
        T *= cooling
        step += 1
        
        # Periodic restarts from NEH with perturbation to escape local optima
        if step % 5000 == 0 and time.time() < deadline - 5.0:
            # Perturb best solution
            perturbed = best_perm.copy()
            for _ in range(3):
                new_p = _neighborhood_moves(perturbed, p, rng)
                perturbed = new_p
            ms_p = _makespan(perturbed, p)
            if ms_p < best_ms * 1.2:  # Accept if not too bad
                perm = perturbed
                T = max(T, best_ms * 0.05)
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(len(best_perm) - 1):
                if time.time() > deadline - 1.0:
                    break
                for j in range(i + 1, len(best_perm)):
                    new_perm = best_perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    ms = _makespan(new_perm, p)
                    if ms < best_ms:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        break
                if improved:
                    break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
