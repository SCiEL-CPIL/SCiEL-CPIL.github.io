import time
import random
import numpy as np


def _build_ap_structure(n=256):
    """Precompute all APs as (d, r, indices) tuples for fast evaluation."""
    aps = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            indices = list(range(r, n, d))
            if len(indices) >= 1:
                aps.append((d, r, indices))
    return aps


def _max_ap_discrepancy(signs, n=256, limit=None, ap_structure=None):
    """Compute max absolute signed sum over any arithmetic progression in [0..n-1].
    Uses precomputed AP structure if available for speed.
    """
    if ap_structure is None:
        ap_structure = _build_ap_structure(n)
    
    max_abs = 1
    for d, r, indices in ap_structure:
        min_p = 0
        max_p = 0
        cur = 0
        for idx in indices:
            cur += signs[idx]
            if cur < min_p:
                min_p = cur
            if cur > max_p:
                max_p = cur
        disc = max_p - min_p
        if disc > max_abs:
            max_abs = disc
            if limit is not None and max_abs >= limit:
                return max_abs
    return max_abs


def _find_worst_ap(signs, n=256, ap_structure=None):
    """Find the AP with the largest absolute discrepancy.
    Returns (d, r, disc, best_subarray) where best_subarray is (start_idx, end_idx) in the AP.
    """
    if ap_structure is None:
        ap_structure = _build_ap_structure(n)
    
    worst_disc = 0
    worst_d = 1
    worst_r = 0
    worst_start = 0
    worst_end = 0

    for d, r, indices in ap_structure:
        min_p = 0
        max_p = 0
        cur = 0
        min_p_idx = 0  # position in AP where min prefix occurs
        max_p_idx = 0  # position in AP where max prefix occurs
        ap_pos = 0
        for idx in indices:
            cur += signs[idx]
            if cur < min_p:
                min_p = cur
                min_p_idx = ap_pos
            if cur > max_p:
                max_p = cur
                max_p_idx = ap_pos
            ap_pos += 1
        disc = max_p - min_p
        if disc > worst_disc:
            worst_disc = disc
            worst_d = d
            worst_r = r
            # The worst subarray is between min_p_idx and max_p_idx (whichever comes first)
            if min_p_idx < max_p_idx:
                worst_start = min_p_idx
                worst_end = max_p_idx
            else:
                worst_start = max_p_idx
                worst_end = min_p_idx
    return worst_d, worst_r, worst_disc, worst_start, worst_end


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]


def _van_der_corput(n):
    """Generate van der Corput-like sequence based on bit-reversal."""
    signs = []
    for i in range(n):
        # Use Gray code / bit-reversal pattern
        x = i
        gray = x ^ (x >> 1)
        # Count trailing zeros or use parity of reversed bits
        rev = 0
        tmp = i
        for _ in range(8):  # up to 256
            rev = (rev << 1) | (tmp & 1)
            tmp >>= 1
        signs.append(1 if rev % 2 == 0 else -1)
    return signs


def _rademacher_orthogonal(n, seed=0):
    """Generate sequence using Rademacher functions (Walsh functions)."""
    rng = random.Random(seed)
    signs = []
    for i in range(n):
        # Use multiple Walsh functions combined
        val = 0
        for k in range(1, 5):
            # Rademacher function r_k(i) = (-1)^(floor(i/2^(k-1)) mod 2)
            r = 1 if (i >> (k-1)) % 2 == 0 else -1
            val += r
        signs.append(1 if val >= 0 else -1)
    return signs


def _local_search(signs, deadline, rng, n=256, ap_structure=None):
    """Simulated annealing local search with fast evaluation and multi-flip moves."""
    if ap_structure is None:
        ap_structure = _build_ap_structure(n)
    
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n, ap_structure=ap_structure)
    best_signs = current_signs[:]
    best_val = current_val

    temp = 2.0
    cool_rate = 0.999
    steps = 0
    no_improve_count = 0

    while time.time() < deadline - 2:
        # Choose move type: single flip (80%) or double flip (20%)
        if rng.random() < 0.8:
            idx = rng.randint(0, n - 1)
            old_vals = [current_signs[idx]]
            current_signs[idx] = -old_vals[0]
        else:
            idx1 = rng.randint(0, n - 1)
            idx2 = rng.randint(0, n - 1)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % n
            old_vals = [current_signs[idx1], current_signs[idx2]]
            current_signs[idx1] = -old_vals[0]
            current_signs[idx2] = -old_vals[1]

        new_val = _max_ap_discrepancy(current_signs, n, limit=best_val + 1, ap_structure=ap_structure)

        delta = new_val - current_val
        if delta < 0 or (delta == 0 and rng.random() < 0.5) or \
           (delta > 0 and rng.random() < np.exp(-delta / max(temp, 0.01))):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = current_signs[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            for i, old in enumerate(old_vals):
                if i == 0:
                    current_signs[idx] = old
                elif i == 1:
                    current_signs[idx2] = old
            no_improve_count += 1

        # Adaptive cooling: cool faster if stuck
        if no_improve_count > 500:
            temp *= 0.9
            no_improve_count = 0
        else:
            temp *= cool_rate

        steps += 1

        if best_val == 1:
            break

    return best_signs, best_val, steps


def _greedy_repair(signs, deadline, n=256, max_rounds=50, ap_structure=None):
    """Greedy repair: find worst AP, try flipping elements to reduce its discrepancy."""
    if ap_structure is None:
        ap_structure = _build_ap_structure(n)
    
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n, ap_structure=ap_structure)
    best_signs = current_signs[:]
    best_val = current_val
    rng = random.Random(123)

    for _ in range(max_rounds):
        if time.time() > deadline - 3:
            break

        d, r, disc, start_pos, end_pos = _find_worst_ap(current_signs, n, ap_structure=ap_structure)
        if disc <= best_val:
            break

        # Get the elements of the worst AP
        ap_indices = list(range(r, n, d))
        if len(ap_indices) < 2:
            break

        # Try flipping each element in the worst subarray to see if it helps
        subarray_indices = ap_indices[start_pos:end_pos + 1]

        best_flip = -1
        best_flip_val = current_val

        for idx in subarray_indices:
            current_signs[idx] = -current_signs[idx]
            v = _max_ap_discrepancy(current_signs, n, limit=best_flip_val, ap_structure=ap_structure)
            if v < best_flip_val:
                best_flip_val = v
                best_flip = idx
            current_signs[idx] = -current_signs[idx]  # revert

        if best_flip != -1:
            current_signs[best_flip] = -current_signs[best_flip]
            current_val = best_flip_val
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs[:]
        else:
            # No single flip helps; try random flips in the AP
            improved = False
            for _ in range(10):
                idx = rng.choice(subarray_indices)
                current_signs[idx] = -current_signs[idx]
                v = _max_ap_discrepancy(current_signs, n, limit=best_val + 1, ap_structure=ap_structure)
                if v < best_val:
                    best_val = v
                    best_signs = current_signs[:]
                    improved = True
                else:
                    current_signs[idx] = -current_signs[idx]  # revert
            if not improved:
                break

    return best_signs, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    n = 256
    rng = random.Random(42)

    # Precompute AP structure for faster evaluation
    ap_structure = _build_ap_structure(n)

    # Phase 1: Generate multiple initial candidates
    candidates = []

    # Thue-Morse
    candidates.append(_thue_morse(n))

    # Van der Corput
    candidates.append(_van_der_corput(n))

    # Rademacher orthogonal
    for seed in range(3):
        candidates.append(_rademacher_orthogonal(n, seed=seed))

    # Random initializations
    for seed in range(4):
        rng2 = random.Random(seed * 1000 + 7)
        candidates.append([rng2.choice((-1, 1)) for _ in range(n)])

    # Evaluate candidates
    scored = []
    for c in candidates:
        v = _max_ap_discrepancy(c, n, ap_structure=ap_structure)
        scored.append((v, c))
    scored.sort(key=lambda x: x[0])

    best_signs = scored[0][1][:]
    best_val = scored[0][0]

    # Phase 2: Local search on best candidates
    for v, c in scored[:4]:
        if time.time() > deadline - 12:
            break
        ls_signs, ls_val, _ = _local_search(c, deadline, rng, n, ap_structure=ap_structure)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 3: Greedy repair
    if time.time() < deadline - 8:
        gr_signs, gr_val = _greedy_repair(best_signs, deadline, n, ap_structure=ap_structure)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    # Phase 4: Final polish - more local search
    if time.time() < deadline - 3 and best_val > 1:
        ls_signs, ls_val, _ = _local_search(best_signs, deadline, rng, n, ap_structure=ap_structure)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 5: Another greedy repair pass
    if time.time() < deadline - 1 and best_val > 1:
        gr_signs, gr_val = _greedy_repair(best_signs, deadline, n, ap_structure=ap_structure)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    return best_signs
