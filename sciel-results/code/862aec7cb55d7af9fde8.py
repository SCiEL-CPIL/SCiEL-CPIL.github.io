"""Improved program: randomized multi-restart CSP + local-search refinement for max-length distinct-consecutive-sum sequence."""
import time
import random

DEADLINE_MARGIN = 2.0  # seconds of safety margin before the 90s wall-clock kill

def _valid_extension(values, x, sums):
    """Check if appending x to values keeps all consecutive-block sums distinct.
    Returns (new_sums, ok) where new_sums is the set of all block sums after append."""
    # New blocks ending at x: x, x+values[-1], x+values[-1]+values[-2], ...
    total = x
    if total in sums:
        return None, False
    for a in reversed(values):
        total += a
        if total in sums:
            return None, False
    new_sums = sums | {total}
    return new_sums, True


def _greedy_extend(values, sums, start_val, rng):
    """Greedily try to extend values by adding candidates from start_val upward.
    Returns (final_values, final_sums)."""
    for x in range(start_val, 257):
        new_sums, ok = _valid_extension(values, x, sums)
        if ok:
            values.append(x)
            sums = new_sums
    return values, sums


def _greedy_extend_randomized(values, sums, start_val, rng):
    """Extend by trying candidates in a randomized order (still strictly increasing constraint).
    We must pick the next value > last, so we shuffle candidates and try them in that order,
    but only accept the first valid one to maintain strict increase."""
    if not values:
        start = 1
    else:
        start = values[-1] + 1
    candidates = list(range(start, 257))
    rng.shuffle(candidates)
    for x in candidates:
        new_sums, ok = _valid_extension(values, x, sums)
        if ok:
            values.append(x)
            sums = new_sums
            # Continue extending from here
            return _greedy_extend_randomized(values, sums, x + 1, rng)
    return values, sums


def _try_remove_and_extend(values, sums, rng, max_removals=3):
    """Local search: try removing the last few elements and re-extending with different choices."""
    best_len = len(values)
    best_vals = list(values)
    best_sums = set(sums)
    
    for removals in range(1, max_removals + 1):
        if len(values) < removals:
            break
        # Remove last 'removals' elements
        trial = list(values[:-removals])
        # Recompute sums for the trial prefix
        trial_sums = set()
        n = len(trial)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += trial[j]
                trial_sums.add(s)
        
        # Try to extend with randomized greedy
        extended_vals, extended_sums = _greedy_extend_randomized(trial, trial_sums, trial[-1] + 1 if trial else 1, rng)
        
        if len(extended_vals) > best_len:
            best_len = len(extended_vals)
            best_vals = extended_vals
            best_sums = extended_sums
    
    return best_vals, best_sums


def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # First, run the deterministic greedy to get a baseline
    values, sums = [], set()
    for x in range(1, 257):
        new_sums, ok = _valid_extension(values, x, sums)
        if ok:
            values.append(x)
            sums = new_sums
    best_len = len(values)
    best_result = list(values)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_result
    
    # Now do randomized multi-restart CSP with local search refinement
    rng = random.Random(42)
    
    # Multiple restarts with different random seeds for value ordering
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy construction
        vals, sm = _greedy_extend_randomized([], set(), 1, rng)
        
        if len(vals) > best_len:
            best_len = len(vals)
            best_result = list(vals)
        
        # Local search refinement: try removing last few and re-extending
        if len(vals) >= best_len:
            refined_vals, refined_sums = _try_remove_and_extend(vals, sm, rng, max_removals=4)
            if len(refined_vals) > best_len:
                best_len = len(refined_vals)
                best_result = list(refined_vals)
        
        # Also try local search on the current best
        if time.time() < deadline - 5:
            best_sums = set()
            n = len(best_result)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += best_result[j]
                    best_sums.add(s)
            refined_vals, refined_sums = _try_remove_and_extend(best_result, best_sums, rng, max_removals=3)
            if len(refined_vals) > best_len:
                best_len = len(refined_vals)
                best_result = list(refined_vals)
    
    # Final polish: try one more pass of local search on the best result
    if time.time() < deadline - 2:
        best_sums = set()
        n = len(best_result)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += best_result[j]
                best_sums.add(s)
        refined_vals, refined_sums = _try_remove_and_extend(best_result, best_sums, rng, max_removals=5)
        if len(refined_vals) > best_len:
            best_len = len(refined_vals)
            best_result = list(refined_vals)
    
    return best_result
