"""Combined program: NEH/CDS variants + Variable Neighborhood Search (VNS)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    if n_machines <= 10:
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        return curr_row[-1]

def _completion_matrix(order, P):
    """Return completion-time matrix C where C[i, m] is completion of order[i] on machine m."""
    n_jobs = len(order)
    if n_jobs == 0:
        return np.empty((0, P.shape[1]), dtype=float)
    n_machines = P.shape[1]
    C = np.empty((n_jobs, n_machines), dtype=float)
    first = order[0]
    C[0, 0] = P[first, 0]
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + P[first, m]
    for i in range(1, n_jobs):
        j = order[i]
        C[i, 0] = C[i-1, 0] + P[j, 0]
        for m in range(1, n_machines):
            C[i, m] = max(C[i-1, m], C[i, m-1]) + P[j, m]
    return C

def _makespan_from_C(C):
    return C[-1, -1]

def _neh_heuristic(P, sort_key='total', seed=None):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    if sort_key == 'total':
        totals = P.sum(axis=1)
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'first':
        key_vals = P[:, 0].copy()
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'last':
        key_vals = P[:, -1].copy()
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)

    current_seq = [int(sorted_indices[0])]
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    return current_seq

def _cds_heuristic(P, seed=None):
    """CDS heuristic: starts from last machine, inserts in reverse order."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    # Sort by total processing time (descending)
    totals = P.sum(axis=1)
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.uniform(0, 1e-6, n_jobs)
        totals = totals + noise
    sorted_indices = np.argsort(-totals)

    # CDS: start from the end, insert in reverse order
    # We build the sequence by inserting each job (in reverse sort order)
    # into the best position considering the last machine primarily.
    current_seq = [int(sorted_indices[-1])]
    for k in range(n_jobs - 2, -1, -1):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    return current_seq

def _discrete_pso(order, P, deadline=None, n_particles=15, max_iters=200):
    """
    Discrete Particle Swarm Optimization for flow-shop scheduling.
    Particles represent permutations; velocity is a set of positions to swap
    with the personal/global best, weighted by inertia, cognitive, and social terms.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    rng = random.Random(42)
    
    # Initialize particles: one is the given order, others are random perturbations
    particles = []
    pbests = []
    pbest_ms = []
    
    # First particle is the input order
    p0 = list(order)
    ms0 = _makespan(p0, P)
    particles.append(p0)
    pbests.append(list(p0))
    pbest_ms.append(ms0)
    
    # Other particles: random swaps from the input
    for p in range(1, n_particles):
        if deadline is not None and time.time() > deadline:
            break
        new_p = list(order)
        # Apply 2-3 random swaps
        n_swaps = rng.randint(2, min(4, n_jobs - 1))
        for _ in range(n_swaps):
            i, j = rng.sample(range(n_jobs), 2)
            new_p[i], new_p[j] = new_p[j], new_p[i]
        ms = _makespan(new_p, P)
        particles.append(new_p)
        pbests.append(list(new_p))
        pbest_ms.append(ms)
    
    # Global best
    gb_idx = min(range(len(pbest_ms)), key=lambda i: pbest_ms[i])
    gbest = list(pbests[gb_idx])
    gbest_ms = pbest_ms[gb_idx]
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for p in range(len(particles)):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute velocity: which positions to swap
            # v = w * r1 * v_old + c1 * r2 * (pbest - x) + c2 * r3 * (gbest - x)
            # In discrete form: for each position, decide to swap with pbest or gbest position
            
            x = particles[p]
            pb = pbests[p]
            
            # Build swap set
            swaps = set()
            
            # Cognitive component: swap positions where x differs from pbest
            r2 = rng.random()
            if r2 < c1:
                for i in range(n_jobs):
                    if x[i] != pb[i]:
                        if rng.random() < 0.5:
                            swaps.add(i)
            
            # Social component: swap positions where x differs from gbest
            r3 = rng.random()
            if r3 < c2:
                for i in range(n_jobs):
                    if x[i] != gbest[i]:
                        if rng.random() < 0.5:
                            swaps.add(i)
            
            # Inertia: keep some random swaps from previous velocity (approximated)
            if w > 0:
                r1 = rng.random()
                if r1 < w:
                    # Add a few random positions
                    n_rand = rng.randint(0, 2)
                    for _ in range(n_rand):
                        swaps.add(rng.randint(0, n_jobs - 1))
            
            # Apply swaps: randomly pair up positions in swaps and swap
            swap_list = list(swaps)
            rng.shuffle(swap_list)
            for i in range(0, len(swap_list) - 1, 2):
                j = swap_list[i]
                k = swap_list[i + 1]
                if j != k:
                    x[j], x[k] = x[k], x[j]
            
            # Evaluate new position
            new_ms = _makespan(x, P)
            
            # Update personal best
            if new_ms < pbest_ms[p] - 1e-12:
                pbest_ms[p] = new_ms
                pbests[p] = list(x)
                
                # Update global best
                if new_ms < gbest_ms - 1e-12:
                    gbest_ms = new_ms
                    gbest = list(x)
        
        # Adaptive parameter adjustment
        progress = it / max(1, max_iters)
        w = 0.9 - 0.5 * progress  # decrease inertia
        c1 = 1.0 + 0.5 * progress  # increase cognitive
        c2 = 2.0 - 0.5 * progress  # decrease social
    
    return gbest

def _local_search_vns(order, P, deadline=None):
    """
    Hybrid local search: Or-opt relocations and adjacent swaps.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_ms = _makespan(current_order, P)

    max_iters = 100

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False

        # N1: adjacent swaps
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan(new_order, P)
            if new_ms < current_ms - 1e-12:
                current_order = new_order
                current_ms = new_ms
                improved = True
                break
        if improved:
            continue

        # N2: Or-opt relocations (move block of size 1..3)
        for block_size in [1, 2, 3]:
            if n_jobs < block_size + 1:
                continue
            found = False
            for i in range(n_jobs - block_size + 1):
                if deadline is not None and time.time() > deadline:
                    break
                block = current_order[i:i+block_size]
                remaining = current_order[:i] + current_order[i+block_size:]
                for j in range(len(remaining) + 1):
                    if deadline is not None and time.time() > deadline:
                        break
                    if j == i or (block_size == 1 and j == i + 1) or \
                       (block_size > 1 and j >= i and j < i + block_size):
                        continue
                    new_order = remaining[:j] + block + remaining[j:]
                    new_ms = _makespan(new_order, P)
                    if new_ms < current_ms - 1e-12:
                        current_order = new_order
                        current_ms = new_ms
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not improved:
            break

    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses multiple NEH/CDS variants followed by Variable Neighborhood Search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_perm = None
    best_ms = float('inf')

    # Generate multiple initial solutions from different heuristics
    initial_perms = []

    # NEH variants
    for sort_key in ['total', 'first_last', 'first', 'last']:
        if time.time() > deadline - 20.0:
            break
        perm = _neh_heuristic(P, sort_key=sort_key)
        ms = _makespan(perm, P)
        initial_perms.append((ms, perm))
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # CDS heuristic
    if time.time() < deadline - 20.0:
        perm_cds = _cds_heuristic(P)
        ms_cds = _makespan(perm_cds, P)
        initial_perms.append((ms_cds, perm_cds))
        if ms_cds < best_ms:
            best_ms = ms_cds
            best_perm = perm_cds

    # A few randomized NEH restarts
    rng = random.Random(12345)
    max_restarts = 8
    for r in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        seed = rng.randint(0, 10**9)
        perm_r = _neh_heuristic(P, sort_key='total', seed=seed)
        ms_r = _makespan(perm_r, P)
        initial_perms.append((ms_r, perm_r))
        if ms_r < best_ms:
            best_ms = ms_r
            best_perm = perm_r

    # Sort initial solutions by makespan (best first)
    initial_perms.sort(key=lambda x: x[0])

    # Apply PSO to the top initial solutions
    # We'll spend time on the best ones first, and if time permits, try others
    max_pso_attempts = min(3, len(initial_perms))
    pso_deadline = deadline - 5.0

    for idx in range(max_pso_attempts):
        if time.time() > pso_deadline:
            break
        ms_init, perm_init = initial_perms[idx]
        # If this initial solution is already worse than our best after PSO, skip
        if ms_init > best_ms and idx > 0:
            continue

        # Run PSO with the initial solution as the seed
        pso_perm = _discrete_pso(perm_init, P, deadline=pso_deadline, 
                                  n_particles=12, max_iters=150)
        pso_ms = _makespan(pso_perm, P)

        if pso_ms < best_ms:
            best_ms = pso_ms
            best_perm = pso_perm

    # Polish with local search on the best solution
    if time.time() < deadline - 2.0 and best_perm is not None:
        final_perm = _local_search_vns(best_perm, P, deadline=deadline - 0.5)
        final_ms = _makespan(final_perm, P)
        if final_ms < best_ms:
            best_ms = final_ms
            best_perm = final_perm

    return list(best_perm)

# EVOLVE-BLOCK-END
