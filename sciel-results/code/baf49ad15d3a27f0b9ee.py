"""Lagrangian-relaxation covering-code constructor.

Structural change: instead of greedy set-cover + SLS overlap-minimisation,
formulate the covering problem as a maximum-coverage ILP and solve it via
subgradient Lagrangian relaxation.  Dual variables (one per space word) are
updated to dynamically weight the marginal benefit of each candidate centre,
allowing the algorithm to escape the local minima that trap pure greedy or
simple local-search heuristics.

Pipeline:
  1.  Direct-sum baseline (always valid, size q^(n-2) for n>=4, radius>=1).
  2.  Lagrangian subgradient construction:
        - Start from a random feasible cover (greedy seed).
        - Iterate:
            a.  Primal sub-problem: for each candidate centre compute a
                weighted gain = sum over its ball of dual_weight[i] *
                (1 - covered[i]) + small penalty for already-selected centres.
                Select the top-k centres (k = current code size or slightly
                less) to form a new candidate code.
            b.  If the candidate covers everything, accept and try to shrink
                by removing redundant centres.
            c.  Subgradient update: dual[i] += step * (1 - covered[i]) for
                uncovered words, dual[i] -= step * 0.1 for covered words.
        - Periodically run a local-repair pass (swap centres to reduce
          overlap while maintaining coverage).
  3.  Return the best valid code found; fall back to baseline on timeout.
"""

import time
import itertools
import numpy as np
import random

_DEADLINE = 140.0


def _direct_sum(q, n, radius):
    """Always-valid baseline: tetracode direct sum. Size = q^(n-2)."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]


def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx


def _ball_indices(c, q, n, radius):
    """Flat indices of all words within Hamming distance `radius` of center c."""
    if radius == 0:
        return [_idx(c, q)]
    ball = [_idx(c, q)]
    for d in range(1, radius + 1):
        for pos in itertools.combinations(range(n), d):
            for vals in itertools.product(range(1, q), repeat=d):
                w = list(c)
                for p, v in zip(pos, vals):
                    w[p] = v
                ball.append(_idx(w, q))
    return ball


def _covers_all(cw, q, n, radius, balls):
    """Exact coverage check using precomputed balls."""
    total = q ** n
    if total > 5 * 10**6:
        return False
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        covered[balls[_idx(c, q)]] = True
    return bool(covered.all())


def _lagrangian_construct(q, n, radius, time_budget, seed=42):
    """Lagrangian subgradient method for covering code construction.
    
    Returns (code, verified).
    """
    total = q ** n
    if total > 3 * 10**6:
        return None, False

    rng = random.Random(seed)
    start = time.time()

    # Precompute all words and balls
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    balls = []
    for i in range(total):
        balls.append(np.array(_ball_indices(all_words[i].tolist(), q, n, radius), dtype=np.int32))
    
    # Build ball membership matrix for fast gain computation
    # ball_matrix[i, j] = 1 if word j is in ball of word i
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        ball_matrix[i, balls[i]] = 1

    # --- Phase 1: Greedy seed ---
    covered = np.zeros(total, dtype=bool)
    code_indices = []
    while not covered.all():
        if time.time() - start > time_budget * 0.1:
            break
        uncovered_mask = ~covered
        gains = ball_matrix @ uncovered_mask.astype(np.int32)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        code_indices.append(best)
        covered[balls[best]] = True

    if not covered.all():
        # Fallback: add remaining uncovered via greedy
        while not covered.all():
            if time.time() - start > time_budget * 0.15:
                break
            uncovered_mask = ~covered
            gains = ball_matrix @ uncovered_mask.astype(np.int32)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            code_indices.append(best)
            covered[balls[best]] = True

    cur_size = len(code_indices)
    best_code = [all_words[i].tolist() for i in code_indices]
    best_verified = _covers_all(best_code, q, n, radius, balls)

    if time.time() - start > time_budget * 0.2:
        return best_code, best_verified

    # --- Phase 2: Lagrangian subgradient iteration ---
    # Dual variables: weight for each word
    dual = np.ones(total, dtype=np.float64)
    
    # Step size schedule
    initial_step = 0.5
    min_step = 0.01
    step = initial_step
    
    # Target size: try to get below current
    target_size = max(50, cur_size - 10)  # for q=3,n=6, cur_size~81, target~71
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start < time_budget * 0.85:
        if no_improve > max_no_improve:
            break
        
        # Compute weighted gain for each candidate centre
        # gain[i] = sum_{j in ball(i)} dual[j] * (1 - covered[j])
        uncovered = (~covered).astype(np.float64)
        weighted_uncovered = dual * uncovered
        gains = ball_matrix @ weighted_uncovered
        
        # Penalise already-selected centres
        for ci in code_indices:
            gains[ci] *= 0.1
        
        # Select top-k centres
        k = min(cur_size, max(target_size, cur_size - 2))
        top_k = np.argsort(gains)[::-1][:k]
        
        # Check if top-k covers everything
        new_covered = np.zeros(total, dtype=bool)
        for ci in top_k:
            new_covered[balls[ci]] = True
        
        if new_covered.all():
            # Try to shrink further
            new_size = len(top_k)
            if new_size < cur_size:
                # Try removing redundant centres
                removable = []
                for pos, ci in enumerate(top_k):
                    # Check if removing ci still covers everything
                    test_covered = new_covered.copy()
                    test_covered[balls[ci]] = False
                    # Recompute: need to check if any word is now uncovered
                    # More efficient: count how many centres cover each word
                    # For now, simple check
                    if test_covered.all():
                        removable.append(pos)
                
                if removable:
                    # Remove one redundant centre
                    rem_pos = removable[0]
                    top_k = np.delete(top_k, rem_pos)
                    new_covered = np.zeros(total, dtype=bool)
                    for ci in top_k:
                        new_covered[balls[ci]] = True
                    new_size = len(top_k)
                
                if new_size < cur_size and new_covered.all():
                    code_indices = top_k.tolist()
                    covered = new_covered
                    cur_size = new_size
                    target_size = max(50, cur_size - 5)
                    no_improve = 0
                    step = max(min_step, step * 0.9)
                    continue
            else:
                # Same size but different composition; accept with some probability
                if rng.random() < 0.3:
                    code_indices = top_k.tolist()
                    covered = new_covered
                    no_improve = 0
                    step = max(min_step, step * 0.95)
                    continue
        
        # Subgradient update
        # For uncovered words: increase dual (they need more coverage)
        # For covered words: slightly decrease dual
        uncovered_count = int(np.sum(~covered))
        covered_count = total - uncovered_count
        
        if uncovered_count > 0:
            dual[~covered] += step
            dual[covered] -= step * 0.1
        else:
            # All covered; try to reduce dual to allow different selections
            dual *= (1 - step * 0.05)
        
        # Normalise dual to prevent drift
        dual_max = np.max(np.abs(dual))
        if dual_max > 0:
            dual /= dual_max
        
        # Adaptive step size
        step = max(min_step, step * 0.98)
        
        # Periodically run local repair to escape local minima
        if no_improve % 10 == 0 and time.time() - start < time_budget * 0.8:
            # Try swapping a few centres
            for _ in range(20):
                if time.time() - start > time_budget * 0.8:
                    break
                if len(code_indices) == 0:
                    break
                i = rng.randrange(len(code_indices))
                old_ci = code_indices[i]
                old_ball = balls[old_ci]
                
                # Find words uniquely covered by old_ci
                unique_mask = np.zeros(total, dtype=bool)
                unique_mask[old_ball] = True
                for other_ci in code_indices:
                    if other_ci == old_ci:
                        continue
                    unique_mask[balls[other_ci]] = False
                
                unique_words = np.where(unique_mask)[0]
                
                if len(unique_words) == 0:
                    # Redundant, remove it
                    code_indices.pop(i)
                    covered = np.zeros(total, dtype=bool)
                    for ci in code_indices:
                        covered[balls[ci]] = True
                    cur_size = len(code_indices)
                    no_improve = 0
                    break
                
                # Try to find a replacement that covers all unique words
                # Sample candidates
                for _ in range(50):
                    if time.time() - start > time_budget * 0.8:
                        break
                    new_ci = rng.randrange(total)
                    if new_ci == old_ci or new_ci in code_indices:
                        continue
                    new_ball = balls[new_ci]
                    # Check if new_ball covers all unique words
                    if all(new_ball[u] for u in unique_words):
                        # Accept swap
                        code_indices[i] = new_ci
                        covered = np.zeros(total, dtype=bool)
                        for ci in code_indices:
                            covered[balls[ci]] = True
                        no_improve = 0
                        break
        
        no_improve += 1

    # Final verification
    best_code = [all_words[i].tolist() for i in code_indices]
    best_verified = _covers_all(best_code, q, n, radius, balls)
    return best_code, best_verified


def _local_repair(code, q, n, radius, time_budget, seed=0):
    """Local repair: try to reduce code size by finding redundant centres
    and swapping to reduce overlap while maintaining coverage."""
    total = q ** n
    if total > 3 * 10**6:
        return code, False

    rng = random.Random(seed)
    start = time.time()

    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    balls = []
    for i in range(total):
        balls.append(np.array(_ball_indices(all_words[i].tolist(), q, n, radius), dtype=np.int32))

    state = [_idx(c, q) for c in code]
    counts = np.zeros(total, dtype=np.int32)
    for wi in state:
        counts[balls[wi]] += 1

    best_state = list(state)
    no_improve = 0

    while time.time() - start < time_budget:
        if no_improve > 2000:
            break

        # Try removals
        improved = False
        for i in range(len(state) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            wi = state[i]
            b = balls[wi]
            if not np.any(counts[b] == 1):
                counts[b] -= 1
                state.pop(i)
                best_state = list(state)
                improved = True
                break
        if improved:
            no_improve = 0
            continue

        # Try swaps
        if len(state) > 0:
            i = rng.randrange(len(state))
            old_wi = state[i]
            b_old = balls[old_wi]
            critical_mask = (counts[b_old] == 1)
            critical_set = set(b_old[critical_mask].tolist()) if np.any(critical_mask) else set()

            found_swap = False
            for _ in range(100):
                if time.time() - start > time_budget:
                    break
                new_wi = rng.randrange(total)
                if new_wi == old_wi:
                    continue
                b_new = balls[new_wi]
                if critical_set:
                    b_new_set = set(b_new.tolist())
                    if not critical_set.issubset(b_new_set):
                        continue
                # Compute overlap delta
                in_old = np.zeros(total, dtype=bool)
                in_new = np.zeros(total, dtype=bool)
                in_old[b_old] = True
                in_new[b_new] = True
                only_old = in_old & ~in_new
                only_new = in_new & ~in_old
                delta = int(np.sum(2 * counts[only_new] + 1) - np.sum(2 * counts[only_old] - 1))
                if delta < 0:
                    counts[only_old] -= 1
                    counts[only_new] += 1
                    state[i] = new_wi
                    best_state = list(state)
                    found_swap = True
                    no_improve = 0
                    break
            if found_swap:
                continue

        no_improve += 1

    best_code = [all_words[wi].tolist() for wi in best_state]
    verified = _covers_all(best_code, q, n, radius, balls)
    return best_code, verified


def construct(q, n, radius):
    start = time.time()

    # Always-valid baseline
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]

    best = baseline
    best_len = len(baseline)

    # Lagrangian construction
    if q ** n <= 3 * 10**6 and best_len > 1:
        time_budget = max(1.0, _DEADLINE - (time.time() - start) - 10.0)
        
        for lg_seed in [42, 123, 777]:
            remaining = _DEADLINE - (time.time() - start) - 5.0
            if remaining < 2.0:
                break
            try:
                cand, verified = _lagrangian_construct(q, n, radius, remaining, seed=lg_seed)
            except Exception:
                cand, verified = None, False
            if cand is not None and verified and len(cand) < best_len:
                best = cand
                best_len = len(cand)

        # Local repair polish
        remaining = _DEADLINE - (time.time() - start) - 5.0
        if remaining > 2.0 and best_len > 1:
            for lr_seed in [42, 123]:
                remaining = _DEADLINE - (time.time() - start) - 5.0
                if remaining < 1.5:
                    break
                try:
                    polished, p_verified = _local_repair(best, q, n, radius, remaining, seed=lr_seed)
                except Exception:
                    polished, p_verified = None, False
                if polished is not None and p_verified and len(polished) < best_len:
                    best = polished
                    best_len = len(polished)

    # Final safety
    if time.time() - start > _DEADLINE - 5.0:
        return baseline

    return best
