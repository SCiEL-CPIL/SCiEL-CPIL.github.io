"""Rewritten program for C3-prime task: minimize max|autoconv| / (sum(a))^2.
Uses structural two-block design plus local sign-flip optimization."""

import numpy as np
from numpy.fft import fft, ifft
import time

N = 400

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def _make_two_block(n_pos, n_neg, pos_first=True):
    """Create a sequence with a block of +1s and a block of -1s.
    n_pos + n_neg = N. Returns array of length N."""
    if pos_first:
        x = np.ones(N)
        if n_neg > 0:
            x[N - n_neg:] = -1.0
    else:
        x = -np.ones(N)
        if n_pos > 0:
            x[N - n_pos:] = 1.0
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin (10s buffer)
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Phase 1: Two-block structured candidates
    # The autoconvolution of a two-block sequence has a simple triangular structure.
    # For x = [1]*a + [-1]*b where a + b = N:
    # The autoconvolution is piecewise linear, and the max |autoconv| occurs at 
    # lag 0 (value = a + b = N) or at the transition lags.
    # Actually, let's just evaluate numerically.
    
    for a in range(0, N + 1):
        if time.time() > deadline - 60:
            break
        b = N - a
        x_pos_first = _make_two_block(a, b, pos_first=True)
        try_candidate(x_pos_first)
        x_neg_first = _make_two_block(a, b, pos_first=False)
        try_candidate(x_neg_first)
    
    # Phase 2: Random candidates as initialization for SA
    rng = np.random.default_rng(42)
    for trial in range(200):
        if time.time() > deadline - 40:
            break
        # Random +/-1 with slight bias to avoid zero sum
        p = 0.5 + 0.3 * rng.random()
        x = rng.choice([-1.0, 1.0], size=N, p=[1 - p, p])
        try_candidate(x)
    
    # Phase 3: Simulated Annealing with FFT-based incremental updates
    x = best_x.copy()
    current_score = best_score
    current_sum = np.sum(x)
    
    size = 2 * N - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    
    F = fft(x, size2)
    
    T = 0.05
    T_min = 0.0005
    cooling_rate = 0.9998
    
    sa_best_x = x.copy()
    sa_best_score = current_score
    
    while T > T_min:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            # Flip 1-3 random bits
            n_flips = int(rng.integers(1, 4))
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            
            old_vals = x[flip_idx]
            delta_sum = -2.0 * np.sum(old_vals)
            new_sum = current_sum + delta_sum
            
            if abs(new_sum) < 1e-12:
                continue
            
            x[flip_idx] *= -1.0
            
            # Incremental FFT update
            delta_x = np.zeros(size2)
            delta_x[flip_idx] = -2.0 * old_vals
            F += fft(delta_x, size2)
            
            A = ifft(F * np.conj(F)).real
            m_new = np.max(np.abs(A[:size]))
            score_new = m_new / (new_sum * new_sum)
            
            if score_new < current_score:
                current_score = score_new
                current_sum = new_sum
                if score_new < sa_best_score:
                    sa_best_score = score_new
                    sa_best_x = x.copy()
            else:
                delta_E = score_new - current_score
                accept = np.exp(-delta_E / T) > rng.random()
                if accept:
                    current_score = score_new
                    current_sum = new_sum
                else:
                    x[flip_idx] *= -1.0
                    F -= fft(delta_x, size2)
                    current_sum = new_sum - delta_sum
        
        T *= cooling_rate
    
    if sa_best_score < best_score:
        best_x = sa_best_x
        best_score = sa_best_score
    
    # Phase 4: Hill climbing refinement on best
    x = best_x.copy()
    sc = best_score
    F = fft(x, size2)
    current_sum = np.sum(x)
    
    improved = True
    max_rounds = 20
    round_num = 0
    while improved and round_num < max_rounds and time.time() < deadline - 3:
        improved = False
        round_num += 1
        # Randomize the order of positions to avoid bias
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 3:
                break
            
            old_val = x[i]
            new_val = -old_val
            delta_sum = -2.0 * old_val
            new_sum = current_sum + delta_sum
            
            if abs(new_sum) < 1e-12:
                continue
            
            x[i] = new_val
            
            delta_x = np.zeros(size2)
            delta_x[i] = -2.0 * old_val
            F += fft(delta_x, size2)
            
            A = ifft(F * np.conj(F)).real
            m_new = np.max(np.abs(A[:size]))
            score_new = m_new / (new_sum * new_sum)
            
            if score_new < sc - 1e-12:
                sc = score_new
                current_sum = new_sum
                improved = True
                best_x = x.copy()
                best_score = sc
            else:
                x[i] = old_val
                F -= fft(delta_x, size2)
                current_sum = new_sum - delta_sum
    
    return best_x
