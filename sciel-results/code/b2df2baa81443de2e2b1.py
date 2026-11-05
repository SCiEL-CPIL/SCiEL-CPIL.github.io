"""Additive bases with few representations: local search via swap moves."""
import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
TARGET_SIZE = TARGET_HI - TARGET_LO + 1  # 129 targets
DEADLINE = 85.0  # seconds wall-clock safety margin


def _representation_counts(mask):
    """Return array rep[t] = number of unordered pairs {a,b} in A with a+b = t,
    for t in [0, 256]. Uses convolution of the indicator vector with itself,
    then adjusts for the diagonal (a==a counted once, not twice)."""
    x = mask.astype(np.float64)
    # Full convolution gives ordered counts; for unordered with equal summands:
    # count of pairs {a,b} with a<=b and a+b=t.
    # conv[t] = sum_a x[a]*x[t-a]  (ordered, includes a!=b twice and a==b once)
    # unordered_with_equal = (conv[t] + diag[t]) / 2 where diag[t]=x[t/2] if t even
    conv = np.convolve(x, x, mode='full')  # length 257
    # diag contribution: for even t, x[t//2]
    diag = np.zeros(257, dtype=np.float64)
    even = np.arange(0, 257, 2)
    diag[even] = x[even // 2]
    rep = (conv + diag) / 2.0
    return rep


def _peak_and_counts(mask):
    rep = _representation_counts(mask)
    counts = rep[TARGET_LO:TARGET_HI + 1].astype(np.int64)
    peak = int(np.max(counts))
    return peak, counts


def _is_valid(mask):
    rep = _representation_counts(mask)
    return np.all(rep[TARGET_LO:TARGET_HI + 1] > 0)


def _initial_valid_set(rng):
    """Build a valid set by starting with a guaranteed-valid base and adding random elements."""
    # Guaranteed valid: A = {0, 128} gives sums {0,128,256} -- not enough.
    # Use A = [0,128] ∪ {random extras} until valid.
    # Actually, a simple valid set: A = [0, 128] gives only 0,128,256.
    # Better: A = [0, N] gives all sums 0..256. That's 129 elements.
    # Start with a smaller valid set: A = [0, 64, 128] gives sums 0,64,128,192,256 -- not enough.
    # Let's just start with the full range and then try to remove elements.
    # Or start with a random set of size ~64 and repair.
    
    # Strategy: start with random set of size 64, then add elements to repair coverage.
    mask = np.zeros(N + 1, dtype=np.int64)
    
    # Start with a base that's likely to cover a lot: random 50 elements
    chosen = set(rng.sample(range(N + 1), 50))
    for v in chosen:
        mask[v] = 1
    
    # Repair: while some target is uncovered, add a random element that helps
    for _ in range(200):
        if _is_valid(mask):
            break
        # Find uncovered targets
        rep = _representation_counts(mask)
        uncovered = np.where(rep[TARGET_LO:TARGET_HI + 1] == 0)[0] + TARGET_LO
        if len(uncovered) == 0:
            break
        # Pick a random uncovered target and find a pair that covers it
        t = rng.choice(uncovered)
        # Need a+b = t with a,b in [0,128]
        possible = []
        for a in range(0, min(N, t) + 1):
            b = t - a
            if 0 <= b <= N:
                possible.append((a, b))
        if possible:
            a, b = rng.choice(possible)
            mask[a] = 1
            mask[b] = 1
    # If still not valid, fall back to full range
    if not _is_valid(mask):
        mask = np.ones(N + 1, dtype=np.int64)
    return mask


def construct():
    start_time = time.time()
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Initial valid set
    mask = _initial_valid_set(rng)
    peak, counts = _peak_and_counts(mask)
    best_mask = mask.copy()
    best_peak = peak
    
    # Local search: swap moves (remove one element, add another)
    # Also try single add/remove moves
    
    def try_move(mask, peak, counts, move_type, idx1, idx2):
        """Apply a move and return (new_peak, new_counts, improved)."""
        old_peak = peak
        if move_type == 'swap':
            # Remove idx1, add idx2
            if mask[idx1] == 0 or mask[idx2] == 1:
                return None
            mask[idx1] = 0
            mask[idx2] = 1
            # Check validity: all targets still covered
            rep = _representation_counts(mask)
            if not np.all(rep[TARGET_LO:TARGET_HI + 1] > 0):
                # Revert
                mask[idx1] = 1
                mask[idx2] = 0
                return None
            new_counts = rep[TARGET_LO:TARGET_HI + 1].astype(np.int64)
            new_peak = int(np.max(new_counts))
            return new_peak, new_counts, new_peak < old_peak
        elif move_type == 'remove':
            if mask[idx1] == 0:
                return None
            mask[idx1] = 0
            rep = _representation_counts(mask)
            if not np.all(rep[TARGET_LO:TARGET_HI + 1] > 0):
                mask[idx1] = 1
                return None
            new_counts = rep[TARGET_LO:TARGET_HI + 1].astype(np.int64)
            new_peak = int(np.max(new_counts))
            return new_peak, new_counts, new_peak < old_peak
        elif move_type == 'add':
            if mask[idx1] == 1:
                return None
            mask[idx1] = 1
            rep = _representation_counts(mask)
            new_counts = rep[TARGET_LO:TARGET_HI + 1].astype(np.int64)
            new_peak = int(np.max(new_counts))
            return new_peak, new_counts, new_peak < old_peak
        return None
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < DEADLINE:
        if no_improve_count > max_no_improve:
            # Restart with a different random seed
            mask = _initial_valid_set(rng)
            peak, counts = _peak_and_counts(mask)
            if peak < best_peak:
                best_mask = mask.copy()
                best_peak = peak
            no_improve_count = 0
            continue
        
        # Try random swap moves
        improved = False
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                break
            a = rng.randint(0, N)
            b = rng.randint(0, N)
            if a == b:
                continue
            result = try_move(mask, peak, counts, 'swap', a, b)
            if result is None:
                continue
            new_peak, new_counts, is_improved = result
            if is_improved:
                peak = new_peak
                counts = new_counts
                improved = True
                no_improve_count = 0
                if peak < best_peak:
                    best_peak = peak
                    best_mask = mask.copy()
            else:
                no_improve_count += 1
        
        # Occasionally try remove moves to shrink the set
        if rng.random() < 0.3:
            a = rng.randint(0, N)
            result = try_move(mask, peak, counts, 'remove', a, -1)
            if result is not None:
                new_peak, new_counts, is_improved = result
                if is_improved:
                    peak = new_peak
                    counts = new_counts
                    no_improve_count = 0
                    if peak < best_peak:
                        best_peak = peak
                        best_mask = mask.copy()
                else:
                    no_improve_count += 1
        
        # Occasionally try add moves
        if rng.random() < 0.2:
            a = rng.randint(0, N)
            result = try_move(mask, peak, counts, 'add', a, -1)
            if result is not None:
                new_peak, new_counts, is_improved = result
                if is_improved:
                    peak = new_peak
                    counts = new_counts
                    no_improve_count = 0
                    if peak < best_peak:
                        best_peak = peak
                        best_mask = mask.copy()
                else:
                    no_improve_count += 1
    
    # Return the best set found
    return list(np.where(best_mask > 0)[0])
