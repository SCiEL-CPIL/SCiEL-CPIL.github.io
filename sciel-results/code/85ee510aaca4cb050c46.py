"""Erdos minimum overlap: binary-like step heights with single transition window."""
import time
import numpy as np

N = 100
DEADLINE = time.time() + 110  # Safety margin before 120s limit


def _make_ramp(a, w):
    """Create step heights with a single ramp from 0 to 1 over w steps starting at position a."""
    h = np.zeros(N)
    # Clamp a and w to valid ranges
    a = int(round(a))
    w = int(round(w))
    if w < 1:
        w = 1
    if a < 0:
        a = 0
    if a + w > N:
        w = N - a
        if w < 1:
            w = 1
            a = N - 1
    
    # Ramp: positions a to a+w-1 get values 1/w, 2/w, ..., 1
    for i in range(w):
        h[a + i] = (i + 1) / w
    
    # Positions after ramp get 1
    if a + w < N:
        h[a + w:] = 1.0
    
    return h


def _score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    max_overlap = 0.0
    for k in range(1, N):
        # overlap(k) = sum_i h[i] * (1-h[i-k]) for i >= k
        if k < N:
            overlap = np.dot(h[k:], 1.0 - h[:N-k])
            if overlap > max_overlap:
                max_overlap = overlap
    return max_overlap


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Golden-section search over window width w
    # a is determined by w: a = N/2 - w/2 + 0.5
    # w ranges from 1 to N
    phi = (1.0 - 5**0.5) / 2.0  # negative golden ratio
    
    w_low = 1.0
    w_high = float(N)
    
    w1 = w_high - phi * (w_high - w_low)
    w2 = w_low - phi * (w_high - w_low)
    
    h1 = _make_ramp(N/2 - w1/2 + 0.5, w1)
    h2 = _make_ramp(N/2 - w2/2 + 0.5, w2)
    s1 = _score(h1)
    s2 = _score(h2)
    
    iterations = 0
    max_iter = 50
    
    while time.time() < DEADLINE and iterations < max_iter:
        iterations += 1
        if s1 < s2:
            w_high = w2
            w2 = w1
            h2 = h1
            s2 = s1
            w1 = w_high - phi * (w_high - w_low)
            h1 = _make_ramp(N/2 - w1/2 + 0.5, w1)
            s1 = _score(h1)
        else:
            w_low = w1
            w1 = w2
            h1 = h2
            s1 = s2
            w2 = w_low - phi * (w_high - w_low)
            h2 = _make_ramp(N/2 - w2/2 + 0.5, w2)
            s2 = _score(h2)
        
        # Track best
        if s1 < best_score:
            best_score = s1
            best_h = h1.copy()
        if s2 < best_score:
            best_score = s2
            best_h = h2.copy()
    
    # Final check
    if s1 < best_score:
        best_h = h1.copy()
    if s2 < best_score:
        best_h = h2.copy()
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2) > 1e-10:
        # Adjust: add small correction distributed
        diff = N/2 - current_sum
        best_h = best_h + diff / N
    
    # Clip to [0, 1]
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
