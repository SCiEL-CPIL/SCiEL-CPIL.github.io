"""Third autocorrelation inequality — DWT-inspired Block Flip Annealing."""
import numpy as np
import time

N = 400
DEADLINE = 110.0  # seconds, safety margin below 150s evaluator limit


def _compute_autocorr(h):
    """
    Compute the third-order autocorrelation (autoconvolution of f with f*f).
    For a discrete signal h of length N, C3[k] = sum_i h[i] * h[i+k] * h[i+2k]
    over valid i for each lag k.
    
    The "upper bound on C3" is the maximum absolute value of this triple product sum.
    """
    # Use FFT-based convolution for efficiency
    # f * f = conv(h, h), then C3 = conv(f*f, h)
    h_float = h.astype(np.float64)
    # Zero-pad to avoid circular convolution
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    # autoconv = conv(h, h)
    autoconv_fft = h_fft * h_fft
    # C3 = conv(autoconv, h)
    c3_fft = autoconv_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    # The linear convolution result of length 3N-1; c3_full[:3*N-1] is valid
    # But we only care about lags k=0,1,...,N-1 where all three h[i], h[i+k], h[i+2k] are in range
    # For lag k, the sum is over i=0 to N-1-2k, so valid for k = 0 to (N-1)//2
    # Take the first N elements as a conservative set of lags
    c3 = c3_full[:N]
    return c3


def _score(h):
    """The evaluator measures the raw value: max |C3| (lower is better)."""
    c3 = _compute_autocorr(h)
    return np.max(np.abs(c3))


def _make_sine_pattern(freq, phase, amp=1.0):
    """Generate a signed step function from a sine wave."""
    x = np.linspace(0, 1, N, endpoint=False)
    raw = amp * np.sin(2 * np.pi * freq * x + phase)
    return np.sign(raw)


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing with Block Flips (DWT-inspired multi-resolution moves).
    """
    start_time = time.time()
    
    # Initialization: Start with a random sign sequence or a good sine pattern
    # Sine patterns often provide a better starting point than pure random
    x = np.linspace(0, 1, N, endpoint=False)
    best_freq = 5
    best_phase = 0.0
    best_init_score = np.inf
    best_init_h = np.ones(N)
    
    # Quick init search
    for freq in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32]:
        for phase in np.linspace(0, np.pi, 20, endpoint=False):
            h = _make_sine_pattern(freq, phase)
            s = _score(h)
            if s < best_init_score:
                best_init_score = s
                best_init_h = h.copy()
                best_freq = freq
                best_phase = phase

    h_best = best_init_h.copy()
    score_best = best_init_score
    
    rng = np.random.default_rng(42)
    
    # Multi-round greedy optimization with occasional random restarts
    # Strategy: 
    # 1. Run greedy coordinate descent to local minimum
    # 2. Apply random perturbation (flip a random block)
    # 3. Run greedy again
    # 4. Keep best result found
    
    def greedy_polish(h_work, s_work, max_rounds=100):
        """Greedy coordinate descent: flip single bits if it improves score."""
        for _ in range(max_rounds):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                h_try = h_work.copy()
                h_try[i] *= -1
                s_try = _score(h_try)
                if s_try < s_work:
                    s_work = s_try
                    h_work = h_try
                    improved = True
            if not improved:
                break
        return h_work, s_work
    
    def block_perturb(h, blen, rng):
        """Flip a random block of length blen."""
        h_try = h.copy()
        if blen >= N:
            h_try[:] *= -1
        else:
            start_idx = rng.integers(0, N - blen + 1)
            h_try[start_idx:start_idx + blen] *= -1
        return h_try
    
    # Phase 1: Greedy polish from the best initialization
    h_work, s_work = greedy_polish(h_best.copy(), score_best)
    if s_work < score_best:
        score_best = s_work
        h_best = h_work.copy()
    
    # Phase 2: Iterative perturbation + greedy refinement
    # Use progressively smaller perturbation blocks as we refine
    perturb_schedules = [
        (64, 10),   # 10 iterations with block size 64
        (32, 15),   # 15 iterations with block size 32
        (16, 20),   # 20 iterations with block size 16
        (8, 30),    # 30 iterations with block size 8
        (4, 40),    # 40 iterations with block size 4
        (2, 50),    # 50 iterations with block size 2
    ]
    
    for blen, n_iter in perturb_schedules:
        if (time.time() - start_time) > DEADLINE - 5:
            break
        for _ in range(n_iter):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Perturb
            h_pert = block_perturb(h_best, blen, rng)
            s_pert = _score(h_pert)
            
            # Greedy polish from perturbed state
            h_pol, s_pol = greedy_polish(h_pert, s_pert, max_rounds=50)
            
            # Keep if better
            if s_pol < score_best:
                score_best = s_pol
                h_best = h_pol.copy()
    
    # Final greedy polish to squeeze out last improvements
    h_best, score_best = greedy_polish(h_best.copy(), score_best, max_rounds=200)
        
    # Final Greedy Polish: Try flipping single bits to see if we can improve further
    # This is fast and often helps clean up the SA result.
    h_work = h_best.copy()
    s_work = score_best
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            h_try = h_work.copy()
            h_try[i] *= -1
            s_try = _score(h_try)
            if s_try < s_work:
                s_work = s_try
                h_work = h_try
                h_best = h_work.copy()
                score_best = s_work
                improved = True

    return h_best.astype(np.float64)


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
