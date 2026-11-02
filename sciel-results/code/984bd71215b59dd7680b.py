import time
import random
import copy
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, strictly increasing, with all positive differences distinct
    across all rows. Minimize the largest entry.
    """
    deadline = time.time() + 112.0
    rng = random.Random(42)

    # Lower bound on M
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1

    best_rows = None
    best_score = None

    # Phase 1: Try to build a valid initial solution using randomized greedy
    # with multiple strategies, then optimize via simulated annealing.
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Try different M targets
        M_target = M_lower + rng.randint(0, 15)
        
        rows = _randomized_greedy_build(n_rows, k, M_target, rng)
        if rows is not None:
            score = _max_entry(rows)
            if best_score is None or score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
                if best_score == M_lower:
                    return best_rows

        # If we have a good solution, spend remaining time on SA optimization
        if best_rows is not None and time.time() < deadline - 8.0:
            optimized = _simulated_annealing(best_rows, n_rows, k, rng, 
                                              time.time() + min(deadline - 2.0, time.time() + 30.0))
            if optimized is not None:
                opt_score = _max_entry(optimized)
                if opt_score < best_score:
                    best_score = opt_score
                    best_rows = optimized
                    if best_score == M_lower:
                        return best_rows

        # If we're running out of time, break
        if time.time() > deadline - 5.0:
            break

        # Vary the random seed for diversity
        rng = random.Random(rng.randint(0, 2**31))

    if best_rows is not None:
        return best_rows

    # Final fallback: pure greedy
    return _greedy_construct(n_rows, k)


def _max_entry(rows):
    return max(max(r) for r in rows)


def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    dset = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            dset.add(row[j] - row[i])
    return dset


def _greedy_construct(n_rows, k):
    """
    Greedy construction: for each row, pick the smallest possible next value
    that doesn't conflict with existing differences.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows


def _randomized_greedy_build(n_rows, k, M_target, rng):
    """
    Build all rows using randomized greedy with bias toward smaller values,
    constrained to stay under M_target where possible.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            # Try values from base+1 up to M_target (or a bit beyond)
            candidates = []
            for x in range(base + 1, M_target + 1):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    candidates.append(x)
            
            if not candidates:
                # Expand search slightly
                for x in range(base + 1, M_target + 20):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                        break
                if not candidates:
                    return None
            
            # Weighted random: smaller values more likely
            if len(candidates) > 1:
                weights = [1.0 / (c - base + 1) ** 0.5 for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                pick = candidates[-1]
                for i, c in enumerate(candidates):
                    cumulative += weights[i]
                    if r <= cumulative:
                        pick = c
                        break
            else:
                pick = candidates[0]
            
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
        
        rows.append(row)
    
    return rows


def _simulated_annealing(rows, n_rows, k, rng, deadline):
    """
    Simulated annealing to minimize the largest entry.
    Moves: pick a random row and a random position, try replacing that element
    with a different value that keeps the row valid and all differences distinct.
    Accept if it improves or with probability exp(-delta/T).
    """
    current = copy.deepcopy(rows)
    current_score = _max_entry(current)
    best = copy.deepcopy(current)
    best_score = current_score
    
    # Precompute used diffs for quick checking
    def get_all_diffs(row_set):
        diffs = set()
        for row in row_set:
            diffs |= _row_diffs(row)
        return diffs
    
    T = max(1.0, current_score * 0.1)
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        # Don't move position 0 (always 0)
        if k + 1 > 1:
            pos = rng.randint(1, k)
        else:
            continue
        
        row = current[row_idx]
        old_val = row[pos]
        
        # Try to find a new value for this position
        # The new value must be > row[pos-1] and < row[pos+1] (if pos+1 < k+1)
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos + 1 < k + 1 else current_score + 10
        
        if hi < lo:
            T *= cooling
            continue
        
        # Sample a few candidate values
        candidates = []
        # Try values near the old value and random values in range
        for _ in range(min(20, hi - lo + 1)):
            if rng.random() < 0.5:
                # Near old value
                delta = rng.randint(-5, 5)
                v = old_val + delta
                if lo <= v <= hi and v != old_val:
                    candidates.append(v)
            else:
                v = rng.randint(lo, hi)
                if v != old_val:
                    candidates.append(v)
        
        # Also try the smallest and largest valid values
        candidates.append(lo)
        candidates.append(hi)
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            
            # Check if replacing row[pos] with new_val keeps row strictly increasing
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos + 1 < k + 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute new diffs for this row
            new_row = row[:]
            new_row[pos] = new_val
            new_row_diffs = _row_diffs(new_row)
            
            # Old diffs for this row
            old_row_diffs = _row_diffs(row)
            
            # Check: new_row_diffs must not conflict with diffs from other rows
            # Diffs that are being removed: old_row_diffs - new_row_diffs
            # Diffs that are being added: new_row_diffs - old_row_diffs
            added = new_row_diffs - old_row_diffs
            removed = old_row_diffs - new_row_diffs
            
            # Check if any added diff conflicts with other rows' diffs
            conflict = False
            for other_idx in range(n_rows):
                if other_idx == row_idx:
                    continue
                other_diffs = _row_diffs(current[other_idx])
                if added & other_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # This move is valid. Compute new score.
            new_score = _max_entry(current[:row_idx] + [new_row] + current[row_idx+1:])
            
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < _exp_neg_delta(delta, T):
                # Accept the move
                row[pos] = new_val
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best = copy.deepcopy(current)
                
                break  # Only one move per iteration
        
        T *= cooling
    
    return best if best_score < current_score else best


def _exp_neg_delta(delta, T):
    """Compute exp(-delta/T) safely."""
    if delta <= 0:
        return 1.0
    if T <= 0:
        return 0.0
    import math
    val = -delta / T
    if val < -50:
        return 0.0
    return math.exp(val)
