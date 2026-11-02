import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _score(h):
    """Discrete score: (sum h_i^2) / ((sum h_i) * max_k conv_k)."""
    h = np.asarray(h, dtype=float)
    s2 = float(np.dot(h, h))
    s1 = float(np.sum(h))
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = float(np.max(conv))
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _neg_score(h):
    return -_score(h)

def _spike_pattern(K, N):
    """Place K spikes at well-separated positions to minimize cross-convolution overlap.
    Uses a Sidon-like spacing: positions chosen so that pairwise differences are distinct
    (or at least large) to keep autoconvolution peak dominated by self-terms."""
    # For well-separated spikes, use positions that are far apart.
    # With N=50, for K spikes, use evenly spaced but with gaps.
    # Key: spikes should be separated by at least 2*max_width to avoid convolution overlap.
    # For single-point spikes, separation >= 1 suffices for no overlap in discrete conv.
    # But to be safe, use larger gaps.
    if K == 1:
        return [N // 2]
    # Use a greedy approach: pick positions maximizing minimum pairwise distance
    # Simple approach: use positions that are roughly N/K apart, with some jitter for Sidon property
    base_gap = N // K
    positions = []
    used = set()
    # Start with evenly spaced
    for i in range(K):
        pos = i * base_gap + (base_gap // 2)
        if pos >= N:
            pos = N - 1 - (K - 1 - i)
        if pos not in used:
            positions.append(pos)
            used.add(pos)
    # If we couldn't place all, fill remaining
    while len(positions) < K:
        for p in range(N):
            if p not in used:
                positions.append(p)
                used.add(p)
                break
    return positions[:K]

def _optimize_spike_heights(positions, h_init=None, maxiter=300):
    """Optimize heights at given spike positions using Nelder-Mead."""
    K = len(positions)
    if K == 0:
        return np.zeros(N), 0.0
    
    def _spike_score(heights):
        h = np.zeros(N)
        for i, p in enumerate(positions):
            h[p] = max(0.0, heights[i])
        return _score(h)
    
    if h_init is None:
        h_init = np.ones(K) * 0.1
    
    def _neg(h):
        return -_spike_score(h)
    
    try:
        res = minimize(_neg, h_init.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
        heights = np.clip(res.x, 0, None)
        h = np.zeros(N)
        for i, p in enumerate(positions):
            h[p] = heights[i]
        return h, _score(h)
    except Exception:
        h = np.zeros(N)
        for i, p in enumerate(positions):
            h[p] = max(0.0, h_init[i])
        return h, _score(h)

def _coordinate_ascent(h0, n_sweeps=300):
    h = np.clip(np.array(h0, dtype=float), 0, None)
    best_s = _score(h)
    for _ in range(n_sweeps):
        if _elapsed() > _DEADLINE:
            break
        improved = False
        for i in range(N):
            if _elapsed() > _DEADLINE:
                break
            base = h[i]
            cands = [0.0]
            if base > 0:
                cands += [base*0.5, base*0.75, base*1.25, base*1.5, base*0.1]
            cands += [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            best_c = base
            best_cs = best_s
            for c in cands:
                h[i] = c
                s = _score(h)
                if s > best_cs:
                    best_cs = s
                    best_c = c
            h[i] = best_c
            if best_cs > best_s:
                best_s = best_cs
                improved = True
        if not improved:
            break
    return h, best_s

def construct():
    global _START
    _START = time.time()

    best_h = np.zeros(N)
    best_s = 0.0

    # Strategy: try multiple spike counts K from 2 to 10
    # For each K, place spikes at well-separated positions and optimize heights
    for K in range(2, 11):
        if _elapsed() > _DEADLINE - 10:
            break
        
        positions = _spike_pattern(K, N)
        
        # Try multiple initializations for spike heights
        for init_type in range(3):
            if _elapsed() > _DEADLINE - 10:
                break
            
            if init_type == 0:
                h_init = np.ones(K) * 0.1
            elif init_type == 1:
                h_init = np.ones(K) * 0.5
            else:
                h_init = np.linspace(0.05, 0.5, K)
            
            h, s = _optimize_spike_heights(positions, h_init, maxiter=200)
            if s > best_s:
                best_s = s
                best_h = h.copy()
        
        # Coordinate ascent refinement
        if _elapsed() < _DEADLINE - 5:
            h_ref, s_ref = _coordinate_ascent(best_h, n_sweeps=100)
            if s_ref > best_s:
                best_s = s_ref
                best_h = h_ref

    # Final polish with Nelder-Mead on full vector
    if _elapsed() < _DEADLINE - 3 and best_s > 0:
        try:
            res = minimize(_neg_score, best_h.copy(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
            h4 = np.clip(res.x, 0, None)
            s4 = _score(h4)
            if s4 > best_s:
                best_s = s4
                best_h = h4
        except Exception:
            pass

    best_h = np.clip(best_h, 0, None)
    return [float(x) for x in best_h]
