import numpy as np
import time

N = 400
DEADLINE = 145.0

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

def _sa_optimize(signs, time_limit):
    """
    Simulated Annealing with multiple move types.
    """
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    # Cooling parameters
    start_temp = 1.0
    end_temp = 0.001
    # We don't know how many iterations we can do, so we estimate based on time.
    # Each iteration is ~1 FFT (fast) + overhead.
    # Let's just run until time limit.
    
    i = 0
    # Adaptive block size: start with small, increase if stuck?
    # Or random choice.
    
    while time.time() - start_time < time_limit:
        i += 1
        # Linear cooling in "iteration space" is hard without knowing total iters.
        # Use time-based cooling? Or just a fixed number of iterations per "phase"?
        # Let's do a simple exponential decay based on i, assuming ~10000 iters in 145s?
        # 145s / 10000 iters = 14.5 ms/iter.
        # FFT of 800 points is ~0.1ms. So we can do ~1000 iters/s.
        # Total iters ~ 100,000.
        
        temp = start_temp * (end_temp / start_temp) ** (i / 100000.0)
        
        # Move selection
        move_type = rng.random()
        if move_type < 0.5:
            # Single flip
            idx = rng.randint(N)
            new_signs = current.copy()
            new_signs[idx] = -new_signs[idx]
        elif move_type < 0.9:
            # Block flip (size 2-16)
            size = rng.randint(2, 17)
            start = rng.randint(0, N - size + 1)
            new_signs = current.copy()
            new_signs[start:start+size] = -new_signs[start:start+size]
        else:
            # Swap two random indices
            i1, i2 = rng.choice(N, 2, replace=False)
            new_signs = current.copy()
            new_signs[i1], new_signs[i2] = new_signs[i2], new_signs[i1]
            
        new_score = _compute_c3_bound(new_signs)
        
        delta = new_score - current_score
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
                
    return best, best_score

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_pattern = np.ones(N)
    best_score = _compute_c3_bound(best_pattern)
    
    # Run SA from several random starts
    num_restarts = 10
    # Allocate time to each restart
    # Leave 5s for final polish
    time_per_restart = (DEADLINE - 5.0) / num_restarts
    
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5.0:
            break
            
        init_signs = rng.choice([1, -1], size=N)
        
        # Run SA for this restart
        optimized, opt_score = _sa_optimize(init_signs, time_per_restart)
        
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
            
    # Final quick polish on the best pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 1.0:
        # A very short SA run or local search
        # Let's do a short SA with low temp to refine
        refined, ref_score = _sa_optimize(best_pattern, DEADLINE - elapsed - 1.0)
        if ref_score < best_score:
            best_score = ref_score
            best_pattern = refined

    result = np.sign(best_pattern).astype(int)
    
    if np.sum(result) == 0:
        result[0] = -result[0]
        
    return result
