import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

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

def _greedy_flip_optimize(signs, max_rounds=50):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Multiple passes until no improvement or max rounds reached.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for _ in range(max_rounds):
        improved = False
        # Try flipping each position
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound(current)
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
    
    # Start with a few diverse initializations
    rng = np.random.RandomState(42)
    
    initializations = []
    # All ones
    initializations.append(np.ones(N))
    # Alternating
    initializations.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    # Random
    initializations.append(rng.choice([1, -1], size=N))
    # Block structure: +1 -1 +1 -1
    p4 = np.ones(N)
    p4[N//4:N//2] = -1
    p4[3*N//4:] = -1
    initializations.append(p4)
    # Sine pattern
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    p5 = np.sign(np.cos(4 * np.pi * x))
    p5[p5 == 0] = 1
    initializations.append(p5)
    
    best_pattern = initializations[0]
    best_score = _compute_c3_bound(best_pattern)
    
    for init in initializations[1:]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        s = _compute_c3_bound(init)
        if s < best_score:
            best_score = s
            best_pattern = init
    
    # Run greedy optimization on the best initial pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        optimized, opt_score = _greedy_flip_optimize(best_pattern, max_rounds=30)
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Try greedy from a second random start if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        rand_start = rng.choice([1, -1], size=N)
        rand_score = _compute_c3_bound(rand_start)
        if rand_score < best_score:
            best_score = rand_score
            best_pattern = rand_start
        optimized2, opt_score2 = _greedy_flip_optimize(best_pattern, max_rounds=20)
        if opt_score2 < best_score:
            best_score = opt_score2
            best_pattern = optimized2
    
    # Ensure we return integer signs
    return np.sign(best_pattern).astype(int)
