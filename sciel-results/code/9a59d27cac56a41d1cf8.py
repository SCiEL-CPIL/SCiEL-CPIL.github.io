import numpy as np
import time

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _compute_c3_bound_fast(signs, dx, n_fft, F_cache):
    """
    Faster version using precomputed FFT size.
    """
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _greedy_flip_optimize(signs, dx, n_fft, max_rounds=50, deadline=140.0):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Multiple passes until no improvement or max rounds reached.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound_fast(current, dx, n_fft, None)
    
    for round_num in range(max_rounds):
        improved = False
        # Try flipping each position
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound_fast(current, dx, n_fft, None)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    
    # Precompute constants
    dx = 0.5 / N
    n_fft = 2 * N
    
    rng = np.random.RandomState(42)
    
    # Strategy: Use sine-based patterns with varying frequency and phase
    # to exploit the spectral properties of the autocorrelation kernel.
    # A low-frequency sine wave creates smooth transitions between +1 and -1 blocks,
    # minimizing high-frequency energy in the autoconvolution.
    
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(N))
    
    # 2. Alternating
    initializations.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    
    # 3. Random
    initializations.append(rng.choice([1, -1], size=N))
    
    # 4. Block structure: +1 -1 +1 -1
    p4 = np.ones(N)
    p4[N//4:N//2] = -1
    p4[3*N//4:] = -1
    initializations.append(p4)
    
    # 5. Sine patterns with different frequencies
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 6, 8, 10]:
        for phase in [0, np.pi/2, np.pi/3, 2*np.pi/3]:
            pattern = np.sign(np.sin(2 * np.pi * freq * x + phase))
            pattern[pattern == 0] = 1
            initializations.append(pattern)
    
    # 6. Cosine patterns
    for freq in [1, 2, 3, 4]:
        for phase in [0, np.pi/4, np.pi/2]:
            pattern = np.sign(np.cos(2 * np.pi * freq * x + phase))
            pattern[pattern == 0] = 1
            initializations.append(pattern)
    
    # 7. Triangular wave patterns
    for freq in [1, 2, 3]:
        tri = 2 * np.abs(2 * (freq * x + 0.25) - np.floor(2 * (freq * x + 0.25) + 0.5)) - 1
        pattern = np.sign(tri)
        pattern[pattern == 0] = 1
        initializations.append(pattern)
    
    # Evaluate all initializations and keep the best
    best_pattern = initializations[0]
    best_score = _compute_c3_bound_fast(best_pattern, dx, n_fft, None)
    
    for init in initializations[1:]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        s = _compute_c3_bound_fast(init, dx, n_fft, None)
        if s < best_score:
            best_score = s
            best_pattern = init
    
    # Run greedy optimization on the best initial pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        optimized, opt_score = _greedy_flip_optimize(best_pattern, dx, n_fft, max_rounds=40, deadline=DEADLINE - 10)
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Try greedy from a few other good initial patterns if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        # Pick the second best initial pattern
        scores = []
        for init in initializations[:10]:
            s = _compute_c3_bound_fast(init, dx, n_fft, None)
            scores.append((s, init))
        scores.sort()
        if len(scores) > 1:
            second_best = scores[1][1]
            if not np.array_equal(second_best, best_pattern):
                optimized2, opt_score2 = _greedy_flip_optimize(second_best, dx, n_fft, max_rounds=20, deadline=DEADLINE - 10)
                if opt_score2 < best_score:
                    best_score = opt_score2
                    best_pattern = optimized2
    
    # Ensure we return integer signs
    return np.sign(best_pattern).astype(int)
