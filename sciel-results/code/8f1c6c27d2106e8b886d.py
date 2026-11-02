<<<<<<< SEARCH
"""Structural rewrite: simulated-annealing stochastic local search with row re-solve, multi-position moves, and parallel restarts."""
import time
import random
import math
=======
"""Structural rewrite: simulated-annealing stochastic local search with row re-solve, multi-position moves, and parallel restarts."""
import time
import random
import math
import sys
>>>>>>> REPLACE
<<<<<<< SEARCH
def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k):
    """Try a single-position move on row ri at index pi.
    Returns (new_row, delta) where delta = new_max - old_max (can be negative, 0, or positive).
    If no valid move found, returns (None, 0)."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    # Candidate range: within [lo, hi], but also allow some expansion
    span = max(1, hi - lo + 1)
    # Try a few random offsets
    candidates = []
    for _ in range(8):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    # Also try a few deterministic near moves
    for d in (1, -1, 2, -2, 3):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    # Pick a random candidate
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    cand_row = row[:]
    cand_row[pi] = nv
    if not _is_sidon(cand_row):
        return None, 0
    new_diffs = _row_diffs_set(cand_row)
    old_diffs = _row_diffs_set(row)
    other_diffs = cur_used - old_diffs
    if new_diffs & other_diffs:
        return None, 0
    # Compute delta in objective (max entry)
    old_max = _max_entry(cur)
    new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max
    return cand_row, delta
=======
def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k):
    """Try a single-position move on row ri at index pi.
    Returns (new_row, delta) where delta = new_max - old_max (can be negative, 0, or positive).
    If no valid move found, returns (None, 0)."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    # Candidate range: within [lo, hi]
    span = max(1, hi - lo + 1)
    # Try a few random offsets
    candidates = []
    for _ in range(12):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    # Also try a few deterministic near moves
    for d in (1, -1, 2, -2, 3, -3, 4, -4):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    # Pick a random candidate
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    cand_row = row[:]
    cand_row[pi] = nv
    if not _is_sidon(cand_row):
        return None, 0
    new_diffs = _row_diffs_set(cand_row)
    old_diffs = _row_diffs_set(row)
    other_diffs = cur_used - old_diffs
    if new_diffs & other_diffs:
        return None, 0
    # Compute delta in objective (max entry)
    old_max = _max_entry(cur)
    new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max
    return cand_row, delta
>>>>>>> REPLACE
<<<<<<< SEARCH
def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    # SA schedule
    T0 = 12.0
    T_end = 0.05
    # We'll run until time_budget expires, with T decaying geometrically
    # Estimate steps per second roughly; we'll just loop and check time
    T = T0
    decay = 0.9995  # per-step decay factor (tuned for a few thousand steps)
    min_T = T_end

    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        # Choose a move type
        r = rng.random()
        if r < 0.45:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is None:
                T = max(min_T, T * decay)
                continue
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(new_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = new_row
                cur_val += delta
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
        elif r < 0.70:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result, delta = _sa_move_row_swap(cur, cur_used, ri, rj, rng, k)
                    if result is not None:
                        if delta <= 0 or rng.random() < math.exp(-delta / T):
                            cur[ri], cur[rj] = result
                            cur_val += delta
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
        elif r < 0.85:
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            # Kick: random restart from best with a small perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            # Perturb a few positions
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else 10**6
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_sidon(cand_row):
                        new_diffs = _row_diffs_set(cand_row)
                        old_diffs = _row_diffs_set(row)
                        other_diffs = cur_used - old_diffs
                        if not (new_diffs & other_diffs):
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)

        T = max(min_T, T * decay)
        # Periodic re-anchoring if we've drifted too far
        if steps % 2000 == 0 and cur_val > best_val + 15:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val

    return best, best_val
=======
def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    # SA schedule with adaptive temperature
    T0 = 15.0
    T_end = 0.02
    T = T0
    decay = 0.9997  # slower decay for more exploration
    min_T = T_end
    no_improve_steps = 0
    max_no_improve = 3000

    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        # Choose a move type
        r = rng.random()
        if r < 0.50:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is None:
                T = max(min_T, T * decay)
                no_improve_steps += 1
                continue
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(new_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = new_row
                cur_val += delta
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
                    no_improve_steps = 0
                    # Slight temperature boost on improvement
                    T = min(T0, T * 1.1)
                else:
                    no_improve_steps += 1
        elif r < 0.72:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result, delta = _sa_move_row_swap(cur, cur_used, ri, rj, rng, k)
                    if result is not None:
                        if delta <= 0 or rng.random() < math.exp(-delta / T):
                            cur[ri], cur[rj] = result
                            cur_val += delta
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
                                no_improve_steps = 0
                                T = min(T0, T * 1.1)
                            else:
                                no_improve_steps += 1
        elif r < 0.88:
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
                        no_improve_steps = 0
                        T = min(T0, T * 1.1)
                    else:
                        no_improve_steps += 1
        else:
            # Kick: random restart from best with a small perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            no_improve_steps = 0
            # Perturb a few positions
            for _ in range(rng.randint(1, 4)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else 10**6
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_sidon(cand_row):
                        new_diffs = _row_diffs_set(cand_row)
                        old_diffs = _row_diffs_set(row)
                        other_diffs = cur_used - old_diffs
                        if not (new_diffs & other_diffs):
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)

        T = max(min_T, T * decay)
        # Periodic re-anchoring if we've drifted too far or stuck
        if steps % 1500 == 0 and (cur_val > best_val + 10 or no_improve_steps > max_no_improve):
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            no_improve_steps = 0
            # Reset temperature on kick
            T = max(T0 * 0.7, T)

    return best, best_val
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Phase 1: diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(8):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    # Pick the best few as starting points for SA
    inits.sort(key=_max_entry)
    top_inits = inits[:4]

    # Phase 2: run multiple SA trajectories in parallel (sequentially, with time budget split)
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 4
    budget_per_sa = max(5.0, remaining / n_sa - 1.0)

    overall_best = None
    overall_best_val = float('inf')
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 2:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Phase 3: final polish with a short SA from the overall best
    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)
    elif time.time() < deadline - 5:
        b, bv = _sa_run(n_rows, k, overall_best, max(3.0, deadline - time.time() - 2), seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
=======
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Phase 1: diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(10):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    # Pick the best few as starting points for SA
    inits.sort(key=_max_entry)
    top_inits = inits[:5]

    # Phase 2: run multiple SA trajectories in parallel (sequentially, with time budget split)
    remaining = deadline - time.time()
    n_sa = 5
    budget_per_sa = max(5.0, remaining / n_sa - 1.0)

    overall_best = None
    overall_best_val = float('inf')
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 3:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Phase 3: final polish with a short SA from the overall best
    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)
    elif time.time() < deadline - 5:
        b, bv = _sa_run(n_rows, k, overall_best, max(3.0, deadline - time.time() - 2), seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
>>>>>>> REPLACE
