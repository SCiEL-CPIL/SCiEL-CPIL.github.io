"""Simulated annealing with vectorized delta evaluation to minimize mono 4-term APs."""
import numpy as np
import time

N = 256

def _precompute_aps():
    """Return (aps_array, pos_to_aps) where aps_array is shape (M,4) of 0-based indices,
    and pos_to_aps maps each position to list of AP indices containing it."""
    aps = []
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            # 1-based: a, a+d, a+2d, a+3d
            # 0-based: a-1, a+d-1, a+2d-1, a+3d-1
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    aps_array = np.array(aps, dtype=np.int32)
    M = len(aps_array)
    
    # Build position-to-AP-index mapping
    pos_to_aps = [[] for _ in range(N)]
    for idx in range(M):
        for pos in aps_array[idx]:
            pos_to_aps[pos].append(idx)
    # Convert to arrays for speed
    pos_to_aps_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_to_aps]
    return aps_array, pos_to_aps_arr

_AP_ARRAY, _POS_TO_APS = _precompute_aps()
_M = len(_AP_ARRAY)

def _count_mono_aps(colours):
    """Count monochromatic 4-term APs. colours is a numpy array of 0/1 of length 256."""
    # For each AP, check if all 4 entries have the same colour
    # c0 == c1 == c2 == c3
    c0 = _AP_ARRAY[:, 0]
    c1 = _AP_ARRAY[:, 1]
    c2 = _AP_ARRAY[:, 2]
    c3 = _AP_ARRAY[:, 3]
    v0 = colours[c0]
    v1 = colours[c1]
    v2 = colours[c2]
    v3 = colours[c3]
    mono = (v0 == v1) & (v1 == v2) & (v2 == v3)
    return int(np.sum(mono))

def _delta_flip(colours, p):
    """Compute the change in mono AP count if we flip position p.
    Returns the delta (new_count - old_count)."""
    # APs that contain position p
    ap_indices = _POS_TO_APS[p]
    if len(ap_indices) == 0:
        return 0
    
    # Get the 4 positions of those APs
    sub_aps = _AP_ARRAY[ap_indices]  # shape (k, 4)
    
    # Get current colours of the 4 positions
    c0 = sub_aps[:, 0]
    c1 = sub_aps[:, 1]
    c2 = sub_aps[:, 2]
    c3 = sub_aps[:, 3]
    
    v0 = colours[c0]
    v1 = colours[c1]
    v2 = colours[c2]
    v3 = colours[c3]
    
    # Currently mono?
    is_mono_before = (v0 == v1) & (v1 == v2) & (v2 == v3)
    
    # After flipping p: the colour at position p changes
    # For each AP, the position p appears in one of the 4 slots
    # Determine which slot p is in for each AP
    # sub_aps has shape (k,4), find where p appears
    # Since p is guaranteed to be in each AP, find the column
    col_idx = np.where(sub_aps == p)
    # col_idx[1] gives the column index for each AP where p appears
    # But we need to be careful: p might appear multiple times? No, in a 4-term AP all positions are distinct.
    # So each AP has p in exactly one position.
    # Build flipped colours: for each AP, flip the value at the position where p is
    flipped_v0 = v0.copy()
    flipped_v1 = v1.copy()
    flipped_v2 = v2.copy()
    flipped_v3 = v3.copy()
    
    # For each AP i, find which slot has p
    # col_idx[0] = row indices, col_idx[1] = column indices
    # Since each row has exactly one p, we can use:
    # For row i, the column is col_idx[1][i]
    # Actually col_idx gives pairs (row, col) where sub_aps[row, col] == p
    # Each row appears exactly once, so:
    # Let's build a mapping
    # Simpler: create a mask for each column
    mask0 = (sub_aps[:, 0] == p)
    mask1 = (sub_aps[:, 1] == p)
    mask2 = (sub_aps[:, 2] == p)
    mask3 = (sub_aps[:, 3] == p)
    
    # Flip the value at position p in the appropriate slot
    # v0 is colours[c0], if c0 == p then flip v0
    flipped_v0[mask0] = 1 - v0[mask0]
    flipped_v1[mask1] = 1 - v1[mask1]
    flipped_v2[mask2] = 1 - v2[mask2]
    flipped_v3[mask3] = 1 - v3[mask3]
    
    is_mono_after = (flipped_v0 == flipped_v1) & (flipped_v1 == flipped_v2) & (flipped_v2 == flipped_v3)
    
    delta = int(np.sum(is_mono_after) - np.sum(is_mono_before))
    return delta

def _sa_single_restart(rng, deadline):
    """Run a single simulated annealing restart. Returns (best_colouring, best_score)."""
    # Initialize randomly
    colours = rng.integers(0, 2, size=N).astype(np.int8)
    current_score = _count_mono_aps(colours)
    best_colouring = colours.copy()
    best_score = current_score
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    # Number of iterations: we'll run until deadline
    # Estimate iterations per second
    t0 = time.time()
    
    # Geometric cooling: T = T_start * (T_end/T_start)^(iter/max_iter)
    # But we don't know max_iter, so use time-based cooling
    # T(t) = T_start * exp(-lambda * t)
    # Choose lambda so that T reaches T_end at ~100s
    lambda_cool = np.log(T_start / T_end) / 100.0
    
    while time.time() < deadline:
        # Pick a random position to flip
        p = int(rng.integers(0, N))
        
        delta = _delta_flip(colours, p)
        
        if delta <= 0:
            # Accept: improves or equal
            colours[p] = 1 - colours[p]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_colouring = colours.copy()
        else:
            # Reject or accept with probability exp(-delta/T)
            elapsed = time.time() - t0
            T = T_start * np.exp(-lambda_cool * elapsed)
            if T > T_end:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    colours[p] = 1 - colours[p]
                    current_score += delta
            else:
                # Temperature too low, stop
                break
    
    return best_colouring, best_score

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    best_colouring = None
    best_score = float('inf')
    
    # First, try a few quick random initializations to get a baseline
    for _ in range(5):
        colours = rng.integers(0, 2, size=N).astype(np.int8)
        score = _count_mono_aps(colours)
        if score < best_score:
            best_score = score
            best_colouring = colours.copy()
    
    # Run simulated annealing restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        # Use the best found so far as a starting point for some restarts,
        # and random for others to maintain diversity
        if r < 3:
            # Start from best
            start_colouring = best_colouring.copy()
        else:
            start_colouring = rng.integers(0, 2, size=N).astype(np.int8)
        
        # Override the initial state in _sa_single_restart by modifying it
        # Actually, let's just run SA from this starting point
        # Modify _sa_single_restart to accept an initial state
        # For simplicity, let's inline the SA loop here with custom start
        
        colours = start_colouring.copy()
        current_score = _count_mono_aps(colours)
        if current_score < best_score:
            best_score = current_score
            best_colouring = colours.copy()
        
        T_start = 5.0
        T_end = 0.01
        t0 = time.time()
        lambda_cool = np.log(T_start / T_end) / 80.0  # Cool over ~80s per restart
        
        while time.time() < deadline:
            p = int(rng.integers(0, N))
            delta = _delta_flip(colours, p)
            
            if delta <= 0:
                colours[p] = 1 - colours[p]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colouring = colours.copy()
            else:
                elapsed = time.time() - t0
                T = T_start * np.exp(-lambda_cool * elapsed)
                if T > T_end:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        colours[p] = 1 - colours[p]
                        current_score += delta
                else:
                    break
    
    # Return as list of ints
    return [int(c) for c in best_colouring]
