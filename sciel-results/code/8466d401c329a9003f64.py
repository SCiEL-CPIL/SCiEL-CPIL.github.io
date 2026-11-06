"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _all_aps():
    """Generate all arithmetic progressions (start, step, length) within [1, N].
    Returns a list of (start, step, length) tuples.
    """
    aps = []
    for start in range(1, N + 1):
        for step in range(1, N + 1):
            # length is the maximum L such that start + (L-1)*step <= N
            max_l = (N - start) // step + 1
            if max_l < 1:
                continue
            for L in range(1, max_l + 1):
                aps.append((start, step, L))
    return aps

# Precompute all APs and their index sets for fast evaluation
# For efficiency, we store for each AP the list of 0-based indices
_AP_INDICES = []
for (s, step, L) in _all_aps():
    # 0-based indices: s-1, s-1+step, ... s-1+(L-1)*step
    indices = list(range(s - 1, s - 1 + L * step, step))
    _AP_INDICES.append(indices)

_NUM_APS = len(_AP_INDICES)

def _max_abs_discrepancy(signs):
    """Compute the maximum absolute signed sum over all APs.
    signs is a list/array of -1/+1 of length N.
    """
    signs_arr = np.array(signs, dtype=np.int8)
    max_abs = 0
    for indices in _AP_INDICES:
        s = int(signs_arr[indices].sum())
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = abs_s
    return max_abs

def _local_search(signs, time_limit, rng):
    """Perform local search to minimize max abs AP discrepancy.
    Uses coordinate descent: try flipping each sign and keep if it reduces max discrepancy.
    """
    signs = list(signs)
    best_disc = _max_abs_discrepancy(signs)
    deadline = time.time() + time_limit
    n_improvements = 0
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            # Try flipping sign at position i
            old_sign = signs[i]
            signs[i] = -old_sign
            disc = _max_abs_discrepancy(signs)
            if disc < best_disc:
                best_disc = disc
                n_improvements += 1
                improved = True
            else:
                signs[i] = old_sign
        if not improved:
            # Stuck in local minimum; try random perturbation
            for _ in range(5):
                idx = rng.randint(0, N - 1)
                signs[idx] = -signs[idx]
            disc = _max_abs_discrepancy(signs)
            if disc < best_disc:
                best_disc = disc
            else:
                # Revert perturbation
                signs[idx] = -signs[idx]
            # Try again from perturbed state
            for i in range(N):
                old_sign = signs[i]
                signs[i] = -old_sign
                disc = _max_abs_discrepancy(signs)
                if disc < best_disc:
                    best_disc = disc
                    improved = True
                else:
                    signs[i] = old_sign
            if not improved:
                # Give up on this perturbation strategy, try a larger one
                for _ in range(10):
                    idx = rng.randint(0, N - 1)
                    signs[idx] = -signs[idx]
                disc = _max_abs_discrepancy(signs)
                if disc < best_disc:
                    best_disc = disc
                else:
                    # Revert
                    for _ in range(10):
                        idx = rng.randint(0, N - 1)
                        signs[idx] = -signs[idx]
                break
    
    return signs, best_disc

def construct():
    rng = random.Random(42)
    deadline = time.time() + 80.0  # Safety margin for 90s limit
    
    # Start with a random initialization
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    best_signs = list(signs)
    best_disc = _max_abs_discrepancy(signs)
    
    # Multiple restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        # Random restart
        signs = [rng.choice((-1, 1)) for _ in range(N)]
        # Quick local search with remaining time
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        # Use a fraction of remaining time for this restart
        search_time = min(remaining * 0.7, 20.0)
        signs, disc = _local_search(signs, search_time, rng)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = list(signs)
        
        num_restarts += 1
    
    return best_signs
# EVOLVE-BLOCK-END
