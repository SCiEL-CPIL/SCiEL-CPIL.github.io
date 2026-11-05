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


def _compute_ap_prefix_extrema(signs, n=256, ap_structure=None):
    """Compute max-min prefix sum for each AP. Returns list of (disc, d, r)."""
    if ap_structure is None:
        ap_structure = _build_ap_structure(n)
    
    results = []
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
        results.append((disc, d, r))
    return results


def _max_ap_discrepancy(signs, n=256, limit=None, ap_structure=None):
    """Compute max absolute signed sum over any arithmetic progression in [0..n-1]."""
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
    """Find the AP with the largest absolute discrepancy."""
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
        min_p_idx = 0
        max_p_idx = 0
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


def _local_search(signs, deadline, rng, n=256, ap_structure=None):
    """Local search with fast evaluation and targeted descent on worst APs."""
    if ap_structure is None:
        ap_structure = _build_ap_structure(n)
    
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n, ap_structure=ap_structure)
    best_signs = current_signs[:]
    best_val = current_val

    temp = 1.0
    no_improve_count = 0

    while time.time() < deadline - 2:
        # Find worst AP and try to fix it directly
        d, r, disc, start_pos, end_pos = _find_worst_ap(current_signs, n, ap_structure=ap_structure)
        
        if disc <= best_val:
            # No AP worse than best, try random improvements
            idx = rng.randint(0, n - 1)
            old_val = current_signs[idx]
            current_signs[idx] = -old_val
            new_val = _max_ap_discrepancy(current_signs, n, limit=best_val + 1, ap_structure=ap_structure)
            
            if new_val < current_val or (new_val == current_val and rng.random() < 0.3):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_signs = current_signs[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                current_signs[idx] = old_val
                no_improve_count += 1
        else:
            # Targeted descent: flip element in worst AP that reduces discrepancy most
            ap_indices = list(range(r, n, d))
            subarray_indices = ap_indices[start_pos:end_pos + 1]
            
            best_flip = -1
            best_flip_val = current_val
            
            for idx in subarray_indices:
                old_val = current_signs[idx]
                current_signs[idx] = -old_val
                v = _max_ap_discrepancy(current_signs, n, limit=best_flip_val, ap_structure=ap_structure)
                if v < best_flip_val:
                    best_flip_val = v
                    best_flip = idx
                current_signs[idx] = old_val
            
            if best_flip != -1:
                current_signs[best_flip] = -current_signs[best_flip]
                current_val = best_flip_val
                if current_val < best_val:
                    best_val = current_val
                    best_signs = current_signs[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No single flip helps; try random flips in the AP
                improved = False
                for _ in range(5):
                    idx = rng.choice(subarray_indices)
                    old_val = current_signs[idx]
                    current_signs[idx] = -old_val
                    v = _max_ap_discrepancy(current_signs, n, limit=best_val + 1, ap_structure=ap_structure)
                    if v < best_val:
                        best_val = v
                        best_signs = current_signs[:]
                        current_val = v
                        improved = True
                    else:
                        current_signs[idx] = old_val
                if not improved:
                    no_improve_count += 1
                    # If stuck, do a random kick
                    if no_improve_count > 100:
                        for _ in range(3):
                            idx = rng.randint(0, n - 1)
                            current_signs[idx] = -current_signs[idx]
                        current_val = _max_ap_discrepancy(current_signs, n, ap_structure=ap_structure)
                        no_improve_count = 0

        if best_val == 1:
            break

    return best_signs, best_val


def _greedy_repair(signs, deadline, n=256, max_rounds=30, ap_structure=None):
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

        ap_indices = list(range(r, n, d))
        if len(ap_indices) < 2:
            break

        subarray_indices = ap_indices[start_pos:end_pos + 1]

        best_flip = -1
        best_flip_val = current_val

        for idx in subarray_indices:
            current_signs[idx] = -current_signs[idx]
            v = _max_ap_discrepancy(current_signs, n, limit=best_flip_val, ap_structure=ap_structure)
            if v < best_flip_val:
                best_flip_val = v
                best_flip = idx
            current_signs[idx] = -current_signs[idx]

        if best_flip != -1:
            current_signs[best_flip] = -current_signs[best_flip]
            current_val = best_flip_val
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs[:]
        else:
            break

    return best_signs, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 256
    rng = random.Random(42)

    ap_structure = _build_ap_structure(n)

    # Phase 1: Generate multiple initial candidates
    candidates = []
    candidates.append(_thue_morse(n))
    
    for seed in range(6):
        rng2 = random.Random(seed * 1000 + 7)
        candidates.append([rng2.choice((-1, 1)) for _ in range(n)])

    scored = []
    for c in candidates:
        v = _max_ap_discrepancy(c, n, ap_structure=ap_structure)
        scored.append((v, c))
    scored.sort(key=lambda x: x[0])

    best_signs = scored[0][1][:]
    best_val = scored[0][0]

    # Phase 2: Local search on best candidates
    for v, c in scored[:3]:
        if time.time() > deadline - 10:
            break
        ls_signs, ls_val = _local_search(c, deadline, rng, n, ap_structure=ap_structure)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 3: Greedy repair
    if time.time() < deadline - 5 and best_val > 1:
        gr_signs, gr_val = _greedy_repair(best_signs, deadline, n, ap_structure=ap_structure)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    # Phase 4: Final polish
    if time.time() < deadline - 2 and best_val > 1:
        ls_signs, ls_val = _local_search(best_signs, deadline, rng, n, ap_structure=ap_structure)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    return best_signs
