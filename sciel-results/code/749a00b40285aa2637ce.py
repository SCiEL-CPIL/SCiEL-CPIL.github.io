"""Rewritten flow-shop heuristic: NEH + fast 2-opt and insertion local search, plus SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a permutation of job indices."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0

    prev = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            # vectorize a little: cur[m] = max(prev[m], cur[m-1]) + pt[m]
            # still sequential in m, but with numpy scalars it's okay
            for m in range(1, n_machines):
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
        prev = cur
    return float(prev[-1])


def _prefix_completions(perm, processing_times):
    """Return list of n_jobs+1 completion vectors for prefixes of perm."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix = [None] * (n_jobs + 1)
    prefix[0] = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        prev = prefix[j]
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
        prefix[j + 1] = cur
    return prefix


def _swap_makespan(perm, i, prefix, processing_times):
    """Makespan after swapping perm[i] and perm[i+1] using prefix[i]."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C_prev = prefix[i]
    pt_b = processing_times[perm[i + 1]]
    pt_a = processing_times[perm[i]]

    # job b first
    C_b = np.empty(n_machines, dtype=np.float64)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = (C_prev[m] if C_prev[m] > C_b[m - 1] else C_b[m - 1]) + pt_b[m]

    # job a second
    C_a = np.empty(n_machines, dtype=np.float64)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = (C_b[m] if C_b[m] > C_a[m - 1] else C_a[m - 1]) + pt_a[m]

    # continue with remaining jobs
    cur = C_a
    for j in range(i + 2, n_jobs):
        pt = processing_times[perm[j]]
        nxt = np.empty(n_machines, dtype=np.float64)
        nxt[0] = cur[0] + pt[0]
        for m in range(1, n_machines):
            nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
        cur = nxt
    return float(cur[-1])


def _eval_insert_incremental(perm, i, j, prefix, processing_times):
    """Incremental makespan of moving perm[i] to position j (0-indexed, j is target index in new list).
    prefix[k] = completion vector after first k jobs of perm.
    If i < j: remove job at i, insert at j (shifts i+1..j-1 left by 1, job goes to j)
    If i > j: remove job at i, insert at j (shifts j..i-1 right by 1, job goes to j)
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j or (i == j - 1):
        # No change (or adjacent swap which is handled by 2-opt)
        return float(prefix[n_jobs][-1])
    
    job = perm[i]
    pt = processing_times[job]
    
    if i < j:
        # Remove job at position i, insert at position j
        # New order: perm[0..i-1], perm[i+1..j], job, perm[j+1..]
        # prefix[i] gives completions before job at i
        # We need to process perm[i+1..j-1] starting from prefix[i]
        # Then process job starting from that
        # Then process perm[j..n_jobs-1]
        
        # Phase 1: process jobs i+1 to j-1 (these shift left)
        cur = prefix[i].copy()
        for pos in range(i + 1, j):
            pt_pos = processing_times[perm[pos]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt_pos[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt_pos[m]
            cur = nxt
        
        # Phase 2: process the moved job
        nxt = np.empty(n_machines, dtype=np.float64)
        nxt[0] = cur[0] + pt[0]
        for m in range(1, n_machines):
            nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
        cur = nxt
        
        # Phase 3: process remaining jobs from j+1 to end
        # These start from where they were before (their prefix is prefix[j+1] in old, but now shifted)
        # Actually, jobs j+1..end now start from cur, but in the original they started from prefix[j+1]
        # We need to recompute from job j+1 onward
        for pos in range(j + 1, n_jobs):
            pt_pos = processing_times[perm[pos]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt_pos[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt_pos[m]
            cur = nxt
        
        return float(cur[-1])
    
    else:
        # i > j: Remove job at i, insert at j
        # New order: perm[0..j-1], job, perm[j..i-1], perm[i+1..]
        # Phase 1: process job starting from prefix[j]
        cur = prefix[j].copy()
        nxt = np.empty(n_machines, dtype=np.float64)
        nxt[0] = cur[0] + pt[0]
        for m in range(1, n_machines):
            nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
        cur = nxt
        
        # Phase 2: process jobs j to i-1 (these shift right)
        for pos in range(j, i):
            pt_pos = processing_times[perm[pos]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt_pos[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt_pos[m]
            cur = nxt
        
        # Phase 3: process remaining jobs from i+1 to end
        for pos in range(i + 1, n_jobs):
            pt_pos = processing_times[perm[pos]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt_pos[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt_pos[m]
            cur = nxt
        
        return float(cur[-1])


def _neh(processing_times, rng):
    """NEH constructive heuristic with incremental insertion evaluation."""
    n_jobs, _ = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))

    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))

    partial = [order[0]]
    for k in range(1, n_jobs):
        job = order[k]
        # Build prefix completions for current partial
        n_partial = len(partial)
        n_machines = processing_times.shape[1]
        prefix = [None] * (n_partial + 1)
        prefix[0] = np.zeros(n_machines, dtype=np.float64)
        for jj in range(n_partial):
            pt = processing_times[partial[jj]]
            prev = prefix[jj]
            cur = np.empty(n_machines, dtype=np.float64)
            cur[0] = prev[0] + pt[0]
            for m in range(1, n_machines):
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
            prefix[jj + 1] = cur
        
        best_pos = 0
        best_ms = float('inf')
        # try each insertion position using incremental eval
        for pos in range(n_partial + 1):
            # Evaluate inserting job at position pos
            # This is equivalent to moving from a virtual position n_partial to pos
            # Use incremental: remove from end (position n_partial), insert at pos
            if pos == n_partial:
                # Just append: compute from prefix[n_partial]
                pt = processing_times[job]
                cur = prefix[n_partial].copy()
                nxt = np.empty(n_machines, dtype=np.float64)
                nxt[0] = cur[0] + pt[0]
                for m in range(1, n_machines):
                    nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
                ms = float(nxt[-1])
            else:
                # Insert at pos: jobs pos..n_partial-1 shift right, job goes at pos
                # Start from prefix[pos], process job, then jobs pos..n_partial-1
                cur = prefix[pos].copy()
                pt = processing_times[job]
                nxt = np.empty(n_machines, dtype=np.float64)
                nxt[0] = cur[0] + pt[0]
                for m in range(1, n_machines):
                    nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
                cur = nxt
                for pp in range(pos, n_partial):
                    pt_pp = processing_times[partial[pp]]
                    nxt = np.empty(n_machines, dtype=np.float64)
                    nxt[0] = cur[0] + pt_pp[0]
                    for m in range(1, n_machines):
                        nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt_pp[m]
                    cur = nxt
                ms = float(cur[-1])
            
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    return partial


def _local_search(perm, processing_times, deadline, rng, n_jobs_cap_insert=200):
    """2-opt + insertion local search with incremental swap and insert eval."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        prefix = _prefix_completions(best_perm, processing_times)
        # 2-opt: scan all adjacent swaps, take best
        best_i = -1
        best_swap_ms = best_ms
        for i in range(n_jobs - 1):
            ms = _swap_makespan(best_perm, i, prefix, processing_times)
            if ms < best_swap_ms - 1e-12:
                best_swap_ms = ms
                best_i = i
        if best_i >= 0:
            best_perm[best_i], best_perm[best_i + 1] = best_perm[best_i + 1], best_perm[best_i]
            best_ms = best_swap_ms
            improved = True
            continue

        # insertion moves using incremental evaluation
        if n_jobs <= n_jobs_cap_insert:
            best_move = None
            best_ins_ms = best_ms
            # randomize start order to avoid deterministic cycling
            idx_order = list(range(n_jobs))
            rng.shuffle(idx_order)
            for i in idx_order:
                if time.time() > deadline - 1.0:
                    break
                job = best_perm[i]
                # positions where inserting job is not same as current
                for j in range(n_jobs):
                    if j == i or j == i + 1:
                        continue
                    ms = _eval_insert_incremental(best_perm, i, j, prefix, processing_times)
                    if ms < best_ins_ms - 1e-12:
                        best_ins_ms = ms
                        best_move = (i, j, ms)
                if best_move is not None:
                    break
            if best_move is not None:
                i, j, ms = best_move
                job = best_perm.pop(i)
                best_perm.insert(j, job)
                best_ms = ms
                improved = True

    return best_perm, best_ms


def _simulated_annealing(perm, ms, processing_times, deadline, rng):
    """SA with 2-opt and insertion moves using incremental evaluation."""
    n_jobs = len(perm)
    cur_perm = list(perm)
    cur_ms = ms
    best_perm = list(cur_perm)
    best_ms = cur_ms

    T = cur_ms * 0.05
    T_min = max(cur_ms * 0.001, 1e-6)
    cool = 0.999

    while time.time() < deadline and T > T_min:
        prefix = _prefix_completions(cur_perm, processing_times)
        if rng.random() < 0.6:
            i = rng.randint(0, n_jobs - 2)
            new_ms = _swap_makespan(cur_perm, i, prefix, processing_times)
            new_perm = list(cur_perm)
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
        else:
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                T *= cool
                continue
            new_ms = _eval_insert_incremental(cur_perm, i, j, prefix, processing_times)
            new_perm = list(cur_perm)
            job = new_perm.pop(i)
            new_perm.insert(j, job)

        delta = new_ms - cur_ms
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur_perm = new_perm
            cur_ms = new_ms
            if cur_ms < best_ms:
                best_ms = cur_ms
                best_perm = list(cur_perm)
        T *= cool

    return best_perm, best_ms


def order_jobs(processing_times):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s

    n_jobs, _ = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    rng = random.Random(123)

    # generate a few diverse starting solutions
    candidates = []
    candidates.append(_neh(processing_times, rng))
    # random NEH with perturbed order
    for _ in range(2):
        rng2 = random.Random(rng.randint(0, 10**9))
        candidates.append(_neh(processing_times, rng2))
    # LPT
    candidates.append(list(np.argsort(-processing_times.sum(axis=1))))

    best_perm = candidates[0]
    best_ms = _makespan(best_perm, processing_times)
    for p in candidates[1:]:
        ms = _makespan(p, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = p

    # local search phase
    if time.time() < deadline - 10.0:
        best_perm, best_ms = _local_search(best_perm, processing_times, deadline - 10.0, rng)

    # simulated annealing if time remains
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _simulated_annealing(best_perm, best_ms, processing_times, deadline - 5.0, rng)

    # final polish with insertion moves too
    if time.time() < deadline - 1.0:
        best_perm, best_ms = _local_search(best_perm, processing_times, deadline - 1.0, rng, n_jobs_cap_insert=200)

    return list(best_perm)
# EVOLVE-BLOCK-END
