"""Erdos minimum overlap - structured optimization approach."""
N = 100

import numpy as np
import time

def _compute_overlap(h, k):
    """Compute overlap integral of h with shifted (1-h) at shift k.
    Overlap = sum_i min(h[i], 1-h[(i-k) % N]) * (2/N)
    """
    shifted_1mh = 1.0 - np.roll(h, k)
    return np.sum(np.minimum(h, shifted_1mh)) * (2.0 / N)

def _max_overlap(h):
    """Maximum overlap over all non-zero shifts."""
    max_ov = 0.0
    for k in range(1, N):
        ov = _compute_overlap(h, k)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust sum to be exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if current_sum == 0:
        return np.full(N, 0.5)
    # Scale proportionally, then clip and iterate
    for _ in range(20):
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if abs(s - target_sum) < 1e-12:
            break
        if s > target_sum:
            # Need to decrease; reduce elements that are > 0.5
            mask = h > 0.5
            if np.any(mask):
                excess = s - target_sum
                n_mask = np.sum(mask)
                h[mask] -= excess / n_mask
            else:
                h -= (s - target_sum) / N
        else:
            # Need to increase; increase elements that are < 0.5
            mask = h < 0.5
            if np.any(mask):
                deficit = target_sum - s
                n_mask = np.sum(mask)
                h[mask] += deficit / n_mask
            else:
                h += (target_sum - s) / N
    h = np.clip(h, 0.0, 1.0)
    # Final exact sum adjustment
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        h[0] += diff
        h[0] = np.clip(h[0], 0.0, 1.0)
    return h

def _make_periodic_pattern(period, phase=0):
    """Create a periodic binary-like pattern with given period.
    Pattern: first ceil(period/2) values near 1, rest near 0.
    """
    h = np.zeros(N)
    half = period // 2
    if period % 2 == 0:
        half = period // 2
    else:
        half = (period + 1) // 2
    for i in range(N):
        pos = (i + phase) % period
        if pos < half:
            h[i] = 1.0
        else:
            h[i] = 0.0
    return _project(h)

def _make_smooth_pattern(freq, amp=1.0):
    """Create a smooth sinusoidal pattern."""
    t = np.arange(N)
    h = 0.5 + amp * 0.5 * np.sin(2 * np.pi * freq * t / N)
    return _project(h)

def _local_search(h, max_iter=2000, step_size=0.1, deadline=None):
    """Simple coordinate descent / local search to minimize max overlap."""
    h = h.copy()
    h = _project(h)
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try perturbing random coordinates
        for _ in range(10):
            idx = np.random.randint(N)
            direction = np.random.choice([-1, 1])
            h_new = h.copy()
            h_new[idx] += direction * step_size
            h_new = _project(h_new)
            score = _max_overlap(h_new)
            if score < best_score - 1e-12:
                h = h_new
                best_h = h_new.copy()
                best_score = score
                improved = True
                break
        if not improved:
            # Try smaller steps
            step_size *= 0.8
            if step_size < 0.001:
                break
    return best_h, best_score

def construct():
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    
    best_h = np.full(N, 0.5)
    best_score = 0.5  # Seed score
    
    # Try various periodic patterns
    for period in range(2, N + 1):
        if time.time() > deadline:
            break
        for phase in range(min(period, 4)):
            h = _make_periodic_pattern(period, phase)
            score = _max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try smooth patterns
    for freq in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        h = _make_smooth_pattern(freq)
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search from best found
    if time.time() < deadline - 10:
        h_ls, score_ls = _local_search(best_h, max_iter=5000, step_size=0.15, deadline=deadline - 5)
        if score_ls < best_score:
            best_score = score_ls
            best_h = h_ls
    
    # Additional local search from random starts if time permits
    if time.time() < deadline - 15:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _project(h_rand)
            h_ls, score_ls = _local_search(h_rand, max_iter=2000, step_size=0.1, deadline=deadline - 5)
            if score_ls < best_score:
                best_score = score_ls
                best_h = h_ls
    
    return best_h
