"""Strategy: Greedy construction + constraint-targeted repair for AP discrepancy."""
import time
import numpy as np

def _build_ap_data(n):
    """
    Precompute all arithmetic progressions.
    Returns:
      progressions: list of index arrays
      index_to_progs: for each position i, list of (prog_id, pos_in_prog)
    """
    progressions = []
    index_to_progs = [[] for _ in range(n)]
    for step in range(1, n + 1):
        for r in range(min(step, n)):
            indices = np.arange(r, n, step, dtype=np.int32)
            if len(indices) > 0:
                pid = len(progressions)
                progressions.append(indices)
                for pos, idx in enumerate(indices):
                    index_to_progs[idx].append((pid, pos))
    return progressions, index_to_progs

def _compute_all_prefix_sums(signs_arr, progressions):
    """Compute prefix sums for all progressions."""
    prefix_sums = []
    for indices in progressions:
        subseq = signs_arr[indices]
        cumsums = np.cumsum(subseq)
        prefix_sums.append(cumsums)
    return prefix_sums

def _get_max_disc_and_worst(prefix_sums):
    """Return (max_disc, list of worst progression IDs)."""
    max_disc = 1
    worst_pids = []
    for pid, ps in enumerate(prefix_sums):
        if len(ps) == 0:
            continue
        m = np.max(np.abs(ps))
        if m > max_disc:
            max_disc = m
            worst_pids = [pid]
        elif m == max_disc and m > 1:
            worst_pids.append(pid)
    return max_disc, worst_pids

def _eval_flip_effect(signs_arr, flip_idx, progressions, index_to_progs, prefix_sums):
    """
    Evaluate the effect of flipping signs_arr[flip_idx] on the max discrepancy.
    Returns the new max_disc if we flip.
    """
    old_val = signs_arr[flip_idx]
    delta = -2 * old_val  # new_val - old_val = -old_val - old_val = -2*old_val

    affected = index_to_progs[flip_idx]
    # Temporarily modify and check
    saved_vals = []
    for pid, pos in affected:
        ps = prefix_sums[pid]
        saved_vals.append(ps[pos:].copy())
        ps[pos:] += delta

    # Compute new max
    new_max = 1
    for pid, ps in enumerate(prefix_sums):
        if len(ps) == 0:
            continue
        m = np.max(np.abs(ps))
        if m > new_max:
            new_max = m

    # Revert
    for pid, pos in affected:
        ps = prefix_sums[pid]
        ps[pos:] = saved_vals[affected.index((pid, pos))] if False else saved_vals[[a for a in affected].index((pid, pos))]

    return new_max

def _eval_flip_effect_fast(signs_arr, flip_idx, progressions, index_to_progs, prefix_sums, prog_max_abs):
    """
    Faster evaluation: only recompute max for affected progressions,
    and track global max via prog_max_abs array.
    Returns new_global_max_disc.
    """
    old_val = signs_arr[flip_idx]
    delta = -2 * old_val

    affected = index_to_progs[flip_idx]
    saved_maxes = []
    for pid, pos in affected:
        ps = prefix_sums[pid]
        saved_maxes.append((pid, pos, prog_max_abs[pid]))
        ps[pos:] += delta
        prog_max_abs[pid] = np.max(np.abs(ps))

    new_global = max(1, int(np.max(prog_max_abs)))

    # Revert
    for pid, pos, old_max in saved_maxes:
        ps = prefix_sums[pid]
        ps[pos:] -= delta
        prog_max_abs[pid] = old_max

    return new_global

def _greedy_construct(n, progressions, index_to_progs, rng, deadline):
    """
    Construct signs greedily: at each position, choose +1 or -1
    to minimize the resulting max discrepancy.
    """
    signs = np.zeros(n, dtype=np.int64)
    # We'll build incrementally. For each position, try both signs and pick the better.
    # But evaluating full max disc at each of 256 positions is expensive.
    # Instead, we use a partial evaluation: only consider progressions that include the new position.
    
    # Actually, for a clean greedy: place signs left to right.
    # At position i, try +1 and -1, compute the max over all progressions 
    # (only those that have been "completed" or partially filled matter).
    # This is complex. Let's use a simpler approach:
    # Initialize with a Legendre-like or Thue-Morse pattern, then repair.
    
    # Thue-Morse: t(i) = (-1)^{popcount(i)}
    for i in range(n):
        signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
    
    return signs

def _repair_phase(signs, n, progressions, index_to_progs, deadline, rng):
    """
    Iterative repair: find worst progressions, try flipping positions within them
    to reduce the global max.
    """
    signs_arr = signs.copy()
    prefix_sums = _compute_all_prefix_sums(signs_arr, progressions)
    prog_max_abs = np.array([np.max(np.abs(ps)) if len(ps) > 0 else 0 for ps in prefix_sums], dtype=np.int64)
    current_disc = max(1, int(np.max(prog_max_abs)))
    best_signs = signs_arr.copy()
    best_disc = current_disc
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb: flip a random few signs to escape local optimum
            for _ in range(rng.randint(2, 5)):
                idx = rng.randint(0, n - 1)
                signs_arr[idx] *= -1
            prefix_sums = _compute_all_prefix_sums(signs_arr, progressions)
            prog_max_abs = np.array([np.max(np.abs(ps)) if len(ps) > 0 else 0 for ps in prefix_sums], dtype=np.int64)
            current_disc = max(1, int(np.max(prog_max_abs)))
            no_improve_count = 0
            continue

        # Find worst progressions
        worst_pids = np.where(prog_max_abs == current_disc)[0]
        if len(worst_pids) == 0:
            break

        # Pick a random worst progression
        pid = int(rng.choice(worst_pids))
        indices = progressions[pid]
        ps = prefix_sums[pid]

        # For each position in this progression, try flipping the sign at that global index
        # and see if it reduces the global max
        best_flip_idx = -1
        best_new_disc = current_disc

        for pos in range(len(indices)):
            flip_idx = int(indices[pos])
            # Evaluate effect of flipping flip_idx
            new_disc = _eval_flip_effect_fast(
                signs_arr, flip_idx, progressions, index_to_progs, prefix_sums, prog_max_abs
            )
            if new_disc < best_new_disc:
                best_new_disc = new_disc
                best_flip_idx = flip_idx

        if best_flip_idx >= 0 and best_new_disc < current_disc:
            # Apply the flip
            old_val = signs_arr[best_flip_idx]
            delta = -2 * old_val
            signs_arr[best_flip_idx] = -old_val
            for pid2, pos2 in index_to_progs[best_flip_idx]:
                prefix_sums[pid2][pos2:] += delta
                prog_max_abs[pid2] = np.max(np.abs(prefix_sums[pid2]))
            current_disc = best_new_disc
            no_improve_count = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs_arr.copy()
        else:
            no_improve_count += 1

    return best_signs, best_disc

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 80  # Safety margin before 90s kill

    progressions, index_to_progs = _build_ap_data(n)

    # Phase 1: Generate multiple initial candidates
    rng = np.random.RandomState(42)

    candidates = []

    # Candidate 1: Thue-Morse
    tm = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int64)
    candidates.append(tm)

    # Candidate 2: Legendre symbol mod a prime near 256
    # Use prime 257: legendre(a, 257) for a=1..256
    p = 257
    def legendre(a, p):
        if a % p == 0:
            return 0
        return pow(a, (p - 1) // 2, p)
    leg = np.array([1 if legendre(i, p) == 1 else -1 for i in range(1, n + 1)], dtype=np.int64)
    candidates.append(leg)

    # Candidate 3: Random
    rand = rng.choice([-1, 1], size=n).astype(np.int64)
    candidates.append(rand)

    # Candidate 4: Alternating with period 4: +1,+1,-1,-1
    alt = np.array([1, 1, -1, -1] * (n // 4), dtype=np.int64)
    candidates.append(alt)

    # Phase 2: For each candidate, do a quick local repair to get a baseline
    best_overall = None
    best_overall_disc = 999

    for ci, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        sub_deadline = min(deadline, t0 + 15 * (ci + 1))  # Allocate time proportionally
        repaired, disc = _repair_phase(cand, n, progressions, index_to_progs, sub_deadline, rng)
        if disc < best_overall_disc:
            best_overall_disc = disc
            best_overall = repaired

    # Phase 3: More intensive repair on the best candidate
    if best_overall is not None and time.time() < deadline:
        final_signs, final_disc = _repair_phase(
            best_overall, n, progressions, index_to_progs, deadline, rng
        )
        if final_disc < best_overall_disc:
            best_overall_disc = final_disc
            best_overall = final_signs

    # Ensure we have a valid result
    if best_overall is None:
        best_overall = candidates[0]

    return list(best_overall.astype(int))
