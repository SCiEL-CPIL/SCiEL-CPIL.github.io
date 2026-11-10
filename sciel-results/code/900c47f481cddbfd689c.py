# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _max_abs_ap_sum(signs):
    """Compute max |sum| over all APs in [0, n-1]."""
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    best = 1
    for d in range(1, n):
        for r in range(d):
            seq = s[r::d]
            if len(seq) > 1:
                cs = np.cumsum(seq)
                m = int(np.max(np.abs(cs)))
                if m > best:
                    best = m
    return best

def _local_search(signs, deadline, rng):
    """Local search: flip bits to reduce max AP discrepancy."""
    n = len(signs)
    signs = list(signs)
    current = _max_abs_ap_sum(signs)
    best_signs = signs[:]
    best_val = current
    start = time.time()
    
    # Precompute for each position, which (d, r) pairs it belongs to
    # For position i, it belongs to AP (r, d) if r = i % d and i < n
    # We'll just re-evaluate fully for simplicity with NumPy speed
    
    step = 0
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        signs[i] = -signs[i]
        new_val = _max_abs_ap_sum(signs)
        if new_val < current:
            current = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = signs[:]
            step = 0
        else:
            signs[i] = -signs[i]  # revert
            step += 1
            if step > 500:
                # Try a few random restarts within local search
                step = 0
                for _ in range(5):
                    i = rng.randint(0, n - 1)
                    signs[i] = -signs[i]
                    new_val = _max_abs_ap_sum(signs)
                    if new_val < current:
                        current = new_val
                        if new_val < best_val:
                            best_val = new_val
                            best_signs = signs[:]
                        break
                    else:
                        signs[i] = -signs[i]
    return best_signs, best_val

def construct():
    deadline = time.time() + 80  # safety margin
    rng = random.Random(42)
    
    # Try multiple random starts
    best_signs = None
    best_val = 10**9
    
    for trial in range(50):
        if time.time() >= deadline:
            break
        signs = [rng.choice((-1, 1)) for _ in range(_N)]
        # Quick local search with limited time
        trial_deadline = min(deadline, time.time() + 1.5)
        res, val = _local_search(signs, trial_deadline, rng)
        if val < best_val:
            best_val = val
            best_signs = res[:]
    
    # Final polish with remaining time
    if best_signs is not None and time.time() < deadline - 2:
        res, val = _local_search(best_signs, deadline, rng)
        if val < best_val:
            best_val = val
            best_signs = res[:]
    
    if best_signs is None:
        best_signs = [1] * _N
    
    return best_signs
# EVOLVE-BLOCK-END
