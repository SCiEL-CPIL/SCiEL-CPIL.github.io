"""Erdos minimum overlap: construct() returns a step-height sequence."""

import numpy as np
import time
from scipy.optimize import minimize

N = 1024  # number of steps (power of 2 for fast FFT)

# EVOLVE-BLOCK-START

def _overlap_score(h):
    """Compute the maximum overlap of h with shifted 1-h, where h is a periodic binary-ish sequence.
    h should be in [0,1] and sum(h) should equal N/2.
    
    The overlap at shift t is sum_i h_i * (1 - h_{(i+t) mod N}).
    We want to minimize the max over t of this sum.
    """
    # For each shift t, compute sum_i h_i * (1 - h_{(i+t) mod N})
    # = sum_i h_i - sum_i h_i * h_{(i+t) mod N}
    # = N/2 - autocorr(h, t)
    
    # So maximizing sum_i h_i * h_{(i+t) mod N} over t minimizes the max overlap.
    # autocorr via FFT
    f = np.fft.fft(h)
    P = f * np.conj(f)
    acorr = np.fft.ifft(P).real
    # acorr[t] = sum_i h_i * h_{(i+t) mod N} for shift t
    # Max overlap = N/2 - min_t acorr[t]... wait, no.
    # overlap(t) = N/2 - acorr(t)
    # We want to minimize max_t overlap(t) = N/2 - min_t acorr(t)
    # So we want to maximize min_t acorr(t)
    min_acorr = np.min(acorr)
    return N / 2.0 - min_acorr


def _gradient_step(h, lr):
    """Simple subgradient-like step for max overlap objective."""
    f = np.fft.fft(h)
    P = f * np.conj(f)
    acorr = np.fft.ifft(P).real
    
    # overlap(t) = N/2 - acorr(t)
    # We want to reduce the maximum overlap. The "hardest" shift is argmin acorr.
    min_idx = np.argmin(acorr)
    
    # Gradient of overlap w.r.t. h: for the minimizing shift t*,
    # d(overlap)/dh_i = -(h_{(i+t*) mod N} - sum_j h_j * delta) ... actually:
    # overlap(t*) = sum_i h_i - sum_i h_i h_{(i+t*) mod N}
    # d(overlap)/dh_i = 1 - h_{(i+t*) mod N}
    # So gradient descent: h_i -= lr * (1 - h_{(i+t*) mod N}) for all i
    
    t_star = min_idx
    shifted_h = np.roll(h, t_star)
    grad = 1.0 - shifted_h
    h_new = h - lr * grad
    # Project to [0,1] and maintain sum = N/2
    h_new = np.clip(h_new, 0.0, 1.0)
    # Adjust to maintain mean
    mean_h = np.mean(h_new)
    h_new = h_new + (0.5 - mean_h)
    h_new = np.clip(h_new, 0.0, 1.0)
    return h_new


def _bang_bang(h, threshold=0.5):
    """Project to bang-bang (0 or 1) while maintaining sum = N/2."""
    # Sort indices by h value, flip the ones that change the sum least
    h_bb = np.zeros_like(h)
    # Find the best threshold for bang-bang that keeps sum close to N/2
    sorted_indices = np.argsort(h)[::-1]  # descending
    best_bb = None
    best_score = float('inf')
    
    # Try various numbers of 1s
    for n_ones in range(max(1, int(N/2) - 10), int(N/2) + 10):
        if n_ones < 1 or n_ones > N - 1:
            continue
        bb = np.zeros(N)
        bb[sorted_indices[:n_ones]] = 1.0
        score = _overlap_score(bb)
        if score < best_score:
            best_score = score
            best_bb = bb.copy()
    
    return best_bb if best_bb is not None else (h > threshold).astype(float)


def _random_binary(seed):
    rng = np.random.RandomState(seed)
    h = rng.randint(0, 2, size=N).astype(float)
    # Ensure exactly N/2 ones
    ones = np.sum(h == 1.0)
    if ones != N / 2:
        indices = np.arange(N)
        if ones > N / 2:
            # Flip some 1s to 0s
            one_idx = np.where(h == 1.0)[0]
            flip = rng.choice(one_idx, size=ones - N // 2, replace=False)
            h[flip] = 0.0
        else:
            zero_idx = np.where(h == 0.0)[0]
            flip = rng.choice(zero_idx, size=N // 2 - ones, replace=False)
            h[flip] = 1.0
    return h


def _fista_optimize(h0, max_iter=500, deadline_margin=20.0):
    """FISTA-style optimization with adaptive step size."""
    h = h0.copy()
    y = h0.copy()
    t = 1.0
    best_h = h0.copy()
    best_score = _overlap_score(h0)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > deadline_margin:
            break
        
        # Gradient step
        f = np.fft.fft(y)
        P = f * np.conj(f)
        acorr = np.fft.ifft(P).real
        min_idx = np.argmin(acorr)
        shifted_y = np.roll(y, min_idx)
        grad = 1.0 - shifted_y
        
        # Adaptive step size (backtracking)
        lr = 0.05
        for _ in range(20):
            h_new = y - lr * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            # Simple check: if score improves, accept
            score_new = _overlap_score(h_new)
            if score_new < best_score:
                break
            lr *= 0.5
        else:
            break
        
        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h + ((t - 1.0) / t_new) * (h - y)
        t = t_new
        
        h = np.clip(h, 0.0, 1.0)
        # Maintain mean
        h = h + (0.5 - np.mean(h))
        h = np.clip(h, 0.0, 1.0)
        
        score = _overlap_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h


def construct():
    """Construct a step-height sequence minimizing max overlap."""
    start = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random binary candidates
    for seed in range(20):
        h_rand = _random_binary(seed)
        candidates.append(h_rand)
    
    # 2. Structured patterns (alternating, etc.)
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # Three-block pattern
    h_3 = np.zeros(N)
    third = N // 3
    h_3[:third] = 1.0
    h_3[2*third:3*third] = 1.0
    # Adjust to get exactly N/2 ones
    ones_3 = np.sum(h_3 == 1.0)
    if ones_3 < N // 2:
        idx = np.where(h_3 == 0.0)[0]
        flip = idx[:N // 2 - ones_3]
        h_3[flip] = 1.0
    elif ones_3 > N // 2:
        idx = np.where(h_3 == 1.0)[0]
        flip = idx[:ones_3 - N // 2]
        h_3[flip] = 0.0
    candidates.append(h_3)
    
    # Evaluate all candidates and pick the best
    best_h = None
    best_score = float('inf')
    for h_cand in candidates:
        score = _overlap_score(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    if best_h is None:
        return np.full(N, 0.5)
    
    # Refine the best candidate with FISTA
    if time.time() - start < deadline:
        h_refined = _fista_optimize(best_h, max_iter=300, deadline_margin=deadline - (time.time() - start))
        score_refined = _overlap_score(h_refined)
        if score_refined < best_score:
            best_h = h_refined
            best_score = score_refined
    
    # Final bang-bang projection to exploit binary structure
    if time.time() - start < deadline:
        h_bb = _bang_bang(best_h)
        score_bb = _overlap_score(h_bb)
        if score_bb < best_score:
            best_h = h_bb
            best_score = score_bb
        elif time.time() - start < deadline:
            # Try FISTA from bang-bang
            h_bb_fista = _fista_optimize(h_bb, max_iter=200, deadline_margin=deadline - (time.time() - start))
            score_bb_fista = _overlap_score(h_bb_fista)
            if score_bb_fista < best_score:
                best_h = h_bb_fista
                best_score = score_bb_fista
    
    # Ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    # Fix sum to exactly N/2
    deficit = N / 2.0 - np.sum(best_h)
    if abs(deficit) > 1e-10:
        # Adjust smallest values first
        sorted_idx = np.argsort(best_h)
        remaining = deficit
        for i in sorted_idx:
            if remaining <= 0:
                break
            delta = min(best_h[i], -remaining) if remaining > 0 else max(best_h[i] - 1.0, -remaining)
            # Actually, if deficit > 0, we need to increase values; if < 0, decrease
            if remaining > 0:
                add = min(1.0 - best_h[i], remaining)
                best_h[i] += add
                remaining -= add
            else:
                rem = -remaining
                sub = min(best_h[i], rem)
                best_h[i] -= sub
                remaining += sub
    
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final safety: if something went wrong, return constant
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h) - N/2.0) > 0.1:
        return np.full(N, 0.5)
    
    return best_h

# EVOLVE-BLOCK-END
