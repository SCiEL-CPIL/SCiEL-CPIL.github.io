import numpy as np
import time

N = 400

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

def _block_flip_move(signs, rng):
    """
    Perform a block flip move: flip a contiguous block of signs.
    Block size is chosen randomly from a distribution.
    """
    # Choose block size: small blocks for fine-tuning, larger for exploration
    block_size = rng.choice([1, 2, 4, 8, 16, 32, 64, 128], p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02])
    if block_size >= N:
        block_size = N // 2
    
    start = rng.randint(0, N - block_size + 1)
    new_signs = signs.copy()
    new_signs[start:start+block_size] *= -1
    return new_signs

def _simulated_annealing(signs, start_temp, end_temp, cooling_rate, max_iter, rng):
    """
    Simulated annealing to minimize the C3 bound.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    temp = start_temp
    for i in range(max_iter):
        if time.time() > 145.0:
            break
        
        new_signs = _block_flip_move(current, rng)
        new_score = _compute_c3_bound(new_signs)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current = new_signs
            current_score = new_score
            
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        
        # Cool down
        temp *= cooling_rate
        if temp < end_temp:
            temp = end_temp
    
    return best, best_score

def _greedy_flip_optimize(signs, max_rounds=20):
    """
    Greedy local search: try flipping individual bits or small blocks to improve.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for _ in range(max_rounds):
        if time.time() > 145.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > 145.0:
                break
            trial = current.copy()
            trial[i] *= -1
            trial_score = _compute_c3_bound(trial)
            if trial_score < current_score:
                current = trial
                current_score = trial_score
                improved = True
        
        if not improved:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Initialize with random signs
    x = rng.choice([-1.0, 1.0], size=N)
    
    best_x = x.copy()
    best_val = _compute_c3_bound(x)
    
    # Phase 1: Simulated annealing with block flips
    # Multiple restarts with different initial temperatures
    sa_configs = [
        (5.0, 0.01, 0.99, 5000),
        (3.0, 0.005, 0.995, 8000),
        (2.0, 0.001, 0.997, 12000),
    ]
    
    for start_temp, end_temp, cooling_rate, max_iter in sa_configs:
        if time.time() > deadline - 30.0:
            break
        
        # Use current best as starting point for next SA round
        sa_result, sa_val = _simulated_annealing(best_x, start_temp, end_temp, cooling_rate, max_iter, rng)
        
        if sa_val < best_val:
            best_val = sa_val
            best_x = sa_result.copy()
    
    # Phase 2: Greedy refinement
    if time.time() < deadline - 10.0:
        greedy_result, greedy_val = _greedy_flip_optimize(best_x, max_rounds=30)
        if greedy_val < best_val:
            best_val = greedy_val
            best_x = greedy_result.copy()
    
    # Phase 3: Final short SA pass for fine-tuning
    if time.time() < deadline - 5.0:
        final_result, final_val = _simulated_annealing(best_x, 0.5, 0.001, 0.99, 3000, rng)
        if final_val < best_val:
            best_val = final_val
            best_x = final_result.copy()
    
    # Ensure output is ±1
    best_x = np.sign(best_x)
    best_x[best_x == 0] = 1.0
    
    return best_x
