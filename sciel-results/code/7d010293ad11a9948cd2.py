"""Rewritten program: iterative local search with efficient AP-sum updates."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
# Precompute all arithmetic progressions as lists of 0-based indices
_AP_CACHE = None

def _get_aps():
    global _AP_CACHE
    if _AP_CACHE is None:
        _AP_CACHE = []
        n = N
        # For each start s (0-based), step d, and length L
        for s in range(n):
            for d in range(1, n - s):
                # max length: floor((n-1 - s)/d) + 1
                max_len = (n - 1 - s) // d + 1
                for L in range(1, max_len + 1):
                    # indices: s, s+d, ..., s+(L-1)*d
                    ap = [s + i * d for i in range(L)]
                    _AP_CACHE.append(ap)
        # Also include singletons explicitly (they are covered above with d=1, L=1, etc.)
        # But ensure every singleton is present
        # Actually singletons are included: s in range(n), d=1, L=1 gives [s]
        # So we're good.
    return _AP_CACHE

def _compute_discrepancy(signs):
    """Compute max absolute sum over all APs."""
    aps = _get_aps()
    max_abs = 0
    for ap in aps:
        s = 0
        for idx in ap:
            s += signs[idx]
        a = abs(s)
        if a > max_abs:
            max_abs = a
    return max_abs

def _compute_ap_sums(signs):
    """Return dict mapping ap_index -> sum, and max_abs."""
    aps = _get_aps()
    sums = [0] * len(aps)
    max_abs = 0
    for i, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += signs[idx]
        sums[i] = s
        a = abs(s)
        if a > max_abs:
            max_abs = a
    return sums, max_abs

def _try_flip(signs, i, sums, ap_index_map):
    """Try flipping sign at position i. Returns (new_max_abs, new_sums) or None if worse."""
    old_sign = signs[i]
    new_sign = -old_sign
    # Update sums for all APs containing index i
    affected = ap_index_map[i]
    max_abs = 0
    new_sums = list(sums)
    for ai in affected:
        ap = _get_aps()[ai]
        # Instead of recomputing, just adjust
        # The sum changes by 2 * new_sign (since we flip from old_sign to new_sign, delta = 2*new_sign)
        # Actually: old contribution was old_sign, new is new_sign, so delta = new_sign - old_sign = 2*new_sign
        delta = 2 * new_sign
        new_sums[ai] = sums[ai] + delta
        a = abs(new_sums[ai])
        if a > max_abs:
            max_abs = a
    # We need to check ALL ap sums, not just affected ones, for the global max
    # But only affected ones changed. The max could be in an unaffected ap.
    # So we need the current max from unaffected + new max from affected.
    # This is expensive to track incrementally. For simplicity, let's just compute full max.
    # Actually, let's just return the affected sums and let the caller decide.
    # For efficiency, let's just do a full recompute of max_abs from new_sums.
    # That's O(num_aps) which is ~50k. Might be okay for limited flips.
    for s_val in new_sums:
        a = abs(s_val)
        if a > max_abs:
            max_abs = a
    return new_sums, max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    random.seed(42)
    
    # Initialize with random signs
    signs = [random.choice((-1, 1)) for _ in range(N)]
    
    # Precompute AP index map: for each position, which APs contain it
    aps = _get_aps()
    ap_index_map = [[] for _ in range(N)]
    for ai, ap in enumerate(aps):
        for idx in ap:
            ap_index_map[idx].append(ai)
    
    # Initial discrepancy
    sums, best_disc = _compute_ap_sums(signs)
    best_signs = list(signs)
    
    # Local search: try flipping each position, keep if improves
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            new_sums, new_disc = _try_flip(signs, i, sums, ap_index_map)
            if new_disc < best_disc:
                signs[i] = -signs[i]
                sums = new_sums
                best_disc = new_disc
                best_signs = list(signs)
                improved = True
            elif new_disc == best_disc:
                # Tie-break: accept with some probability to escape plateaus
                if random.random() < 0.1:
                    signs[i] = -signs[i]
                    sums = new_sums
                    best_signs = list(signs)
                    improved = True
    
    # Simulated annealing style: try random flips with acceptance probability
    current_disc = best_disc
    current_signs = list(best_signs)
    current_sums, _ = _compute_ap_sums(current_signs)
    
    temp = 2.0
    cooling = 0.95
    while temp > 0.01 and (time.time() < deadline - 2):
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            i = random.randrange(N)
            new_sums, new_disc = _try_flip(current_signs, i, current_sums, ap_index_map)
            delta = new_disc - current_disc
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.001)):
                current_signs[i] = -current_signs[i]
                current_sums = new_sums
                current_disc = new_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = list(current_signs)
        temp *= cooling
    
    # Final polish: deterministic local search
    final_signs = list(best_signs)
    final_sums, final_disc = _compute_ap_sums(final_signs)
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            new_sums, new_disc = _try_flip(final_signs, i, final_sums, ap_index_map)
            if new_disc < final_disc:
                final_signs[i] = -final_signs[i]
                final_sums = new_sums
                final_disc = new_disc
                improved = True
    
    return final_signs
# EVOLVE-BLOCK-END
