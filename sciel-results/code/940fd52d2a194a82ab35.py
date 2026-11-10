"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _precompute_aps():
    """
    Precompute all arithmetic progressions and their membership.
    Returns:
        ap_indices: list of lists, each containing indices of one AP
        pos_to_aps: for each position, list of (ap_idx, position_in_ap) tuples
    """
    n = _N
    ap_indices = []
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for r in range(d):
            seq = list(range(r, n, d))
            if len(seq) > 1:
                ap_idx = len(ap_indices)
                ap_indices.append(seq)
                for pos_in_ap, orig_idx in enumerate(seq):
                    pos_to_aps[orig_idx].append((ap_idx, pos_in_ap))
    
    return ap_indices, pos_to_aps

def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _thue_morse_init():
    """
    Generate Thue-Morse sequence: s(n) = (-1)^(popcount(n))
    This has low discrepancy for intervals.
    """
    n = _N
    signs = np.ones(n, dtype=np.int32)
    for i in range(n):
        bits = bin(i).count('1')
        if bits % 2 == 1:
            signs[i] = -1
    return signs

def _local_search_fast(signs_list, ap_indices, pos_to_aps, max_time, rng):
    """
    Fast local search with incremental AP sum updates.
    """
    n = _N
    signs = np.array(signs_list, dtype=np.int32)
    
    # Compute initial AP sums
    num_aps = len(ap_indices)
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for ap_idx, ap in enumerate(ap_indices):
        ap_sums[ap_idx] = np.sum(signs[ap])
    
    current_max_disc = int(np.max(np.abs(ap_sums)))
    if current_max_disc < 1:
        current_max_disc = 1
    best_disc = current_max_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        # Update AP sums for all APs containing position i
        new_max_disc = 1
        for ap_idx, pos_in_ap in pos_to_aps[i]:
            ap_sums[ap_idx] += -2 * old_sign
            abs_sum = abs(ap_sums[ap_idx])
            if abs_sum > new_max_disc:
                new_max_disc = abs_sum
        
        if new_max_disc < current_max_disc:
            current_max_disc = new_max_disc
            if current_max_disc < best_disc:
                best_disc = current_max_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[i] = old_sign
            for ap_idx, pos_in_ap in pos_to_aps[i]:
                ap_sums[ap_idx] += 2 * old_sign
        
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
    
    return best_signs.tolist(), best_disc

def construct():
    """
    Construct 256 signs minimizing max |sum| over all arithmetic progressions.
    Strategy: Thue-Morse init + fast local search with incremental updates.
    """
    start_time = time.time()
    rng = random.Random(42)
    np.random.seed(42)
    
    # Precompute APs
    ap_indices, pos_to_aps = _precompute_aps()
    
    # Phase 1: Thue-Morse initialization
    tm_signs = _thue_morse_init()
    tm_disc = _compute_max_discrepancy(tm_signs)
    
    # Phase 2: Fast local search from Thue-Morse start
    remaining_time = _DEADLINE - 5 - (time.time() - start_time)
    if remaining_time > 1:
        ls_signs, ls_disc = _local_search_fast(tm_signs.tolist(), ap_indices, pos_to_aps, remaining_time, rng)
        if ls_disc <= tm_disc:
            best_signs = ls_signs
            best_disc = ls_disc
        else:
            best_signs = tm_signs.tolist()
            best_disc = tm_disc
    else:
        best_signs = tm_signs.tolist()
        best_disc = tm_disc
    
    # Check deadline
    if time.time() > _DEADLINE - 2:
        return best_signs
    
    # Phase 3: Try a few random restarts if time permits
    for restart in range(5):
        if time.time() > _DEADLINE - 5:
            break
        
        # Random init
        random_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        
        # Short local search
        short_time = min(3.0, _DEADLINE - 5 - (time.time() - start_time))
        if short_time < 0.5:
            break
        
        ls_signs, ls_disc = _local_search_fast(random_signs, ap_indices, pos_to_aps, short_time, rng)
        
        if ls_disc < best_disc:
            best_disc = ls_disc
            best_signs = ls_signs
    
    return best_signs
# EVOLVE-BLOCK-END
