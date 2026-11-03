"""Rewritten program: signed step heights minimizing max |autoconv| / integral^2."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv_max_abs(h):
    """Compute max absolute value of autoconvolution of h."""
    if len(h) == 0:
        return 0.0
    # Linear autoconvolution via FFT for speed
    n = len(h)
    size = 2 * n
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    H = np.fft.rfft(h, n=fft_size)
    AC = np.fft.irfft(H * np.conj(H), n=fft_size)[:2*n-1]
    return np.max(np.abs(AC))

def _score(h):
    """Score = max|autoconv| / integral^2. Lower is better."""
    integral = np.sum(h)
    if abs(integral) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (integral * integral)

def construct():
    """
    Construct signed step heights minimizing max |autoconv| / (sum h)^2.
    
    Strategy:
    1. Start with diverse random initializations (random signs, golden ratio patterns, etc.)
    2. Local search via sign flips and small perturbations
    3. Deadline guard to ensure we return within time limit
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = 1e10
    
    # Helper to update best
    def update_best(h):
        nonlocal best_h, best_score
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # --- Phase 1: Generate diverse initial candidates ---
    rng = np.random.RandomState(42)
    
    # Candidate 1: All ones (baseline)
    h1 = np.ones(N)
    update_best(h1)
    
    # Candidate 2: Random signs
    h2 = np.random.choice([-1, 1], size=N)
    update_best(h2)
    
    # Candidate 3: Golden ratio based pattern
    phi = (1 + np.sqrt(5)) / 2
    h3 = np.sign(np.sin(np.arange(N) * 2 * np.pi / phi))
    h3[h3 == 0] = 1
    update_best(h3)
    
    # Candidate 4: Alternating
    h4 = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    update_best(h4)
    
    # Candidate 5: Block alternating (blocks of 4)
    h5 = np.array([1 if (i // 4) % 2 == 0 else -1 for i in range(N)])
    update_best(h5)
    
    # Candidate 6: Random with more structure - random walk
    steps = rng.choice([-1, 1], size=N)
    h6 = np.sign(np.cumsum(steps))
    h6[h6 == 0] = 1
    update_best(h6)
    
    # Candidate 7: Mersenne Twister random
    rng2 = np.random.RandomState(12345)
    h7 = rng2.choice([-1, 1], size=N)
    update_best(h7)
    
    # Candidate 8: Smooth periodic
    h8 = np.sign(np.sin(2 * np.pi * np.arange(N) / 10))
    h8[h8 == 0] = 1
    update_best(h8)
    
    # Candidate 9: Two-scale pattern
    h9 = np.array([1 if (i % 8) < 4 else -1 for i in range(N)])
    update_best(h9)
    
    # Candidate 10: Random with low frequency bias
    low_freq = np.sin(2 * np.pi * np.arange(N) / N)
    h10 = np.sign(low_freq + 0.1 * rng.randn(N))
    h10[h10 == 0] = 1
    update_best(h10)
    
    # --- Phase 2: Local search from best candidate ---
    # Use the best candidate so far as starting point for local search
    if best_h is not None:
        current = best_h.copy()
        current_score = best_score
        
        # Sign flip local search
        while time.time() < deadline - 30.0:  # Leave 30s for final polish
            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() > deadline - 30.0:
                    break
                current[i] = -current[i]
                new_score = _score(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = -current[i]  # revert
            if not improved:
                break
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_h = current.copy()
        
        # Perturbation-based local search from best
        # Try small random perturbations (flip a few signs)
        best_h_copy = best_h.copy()
        for _ in range(100):  # Up to 100 perturbation attempts
            if time.time() > deadline - 20.0:
                break
            perturbed = best_h_copy.copy()
            # Flip 1-3 random positions
            n_flips = rng.randint(1, 4)
            flip_indices = rng.choice(N, size=n_flips, replace=False)
            perturbed[flip_indices] *= -1
            p_score = _score(perturbed)
            if p_score < best_score:
                best_score = p_score
                best_h = perturbed.copy()
                best_h_copy = perturbed.copy()
    
    # --- Phase 3: Final high-precision polish ---
    # Do a more careful local search with smaller steps
    if best_h is not None and time.time() < deadline - 5.0:
        current = best_h.copy()
        current_score = _score(current)
        
        # Iterated local search with cooling
        for iteration in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                current[i] = -current[i]
                new_score = _score(current)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = -current[i]
            if not improved:
                break
            if current_score < best_score:
                best_score = current_score
                best_h = current.copy()
    
    return best_h
# EVOLVE-BLOCK-END
