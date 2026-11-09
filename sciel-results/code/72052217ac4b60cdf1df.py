import numpy as np
import time

N = 400

def _compute_max_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _compute_sos_metric(h):
    """Compute sum of squared autocorrelation at non-zero lags."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    return np.sum(R[1:]**2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_h = None
    best_max_score = float('inf')
    
    # Multiple restarts of Simulated Annealing
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        h = np.random.choice([-1.0, 1.0], N)
        
        # Phase 1: Simulated Annealing on Sum of Squares (SOS)
        T = 1.0
        cooling_rate = 0.99995
        min_T = 1e-3
        current_sos = _compute_sos_metric(h)
        
        # We need to estimate steps to reach min_T.
        # ln(min_T) / ln(cooling_rate)
        # ln(1e-3) / ln(0.99995) approx -6.9 / -0.00005 = 138000 steps.
        # This is too many if slow. Let's use faster cooling or fewer steps.
        # Let's target ~50,000 steps per restart.
        # 5 restarts * 50k = 250k steps.
        # If 1 step is 0.5ms, 250k * 0.5ms = 125s. Tight.
        # Let's optimize the metric calculation.
        
        # Instead of full FFT every time, we can update autocorrelation incrementally?
        # Flipping h[i] changes R[k] by -2 * h[i] * (h[i-k] + h[i+k])?
        # R[k] = sum h[j] h[j-k].
        # If h[i] -> -h[i], the terms involving h[i] change sign.
        # R_new[k] = R_old[k] - 2*h[i]*(h[i-k] + h[i+k]) for k != 0?
        # Actually, R[k] is circular.
        # This incremental update is $O(N)$ per step, vs $O(N \log N)$ for FFT.
        # For N=400, $N \log N \approx 400 * 9 = 3600$. $O(N) = 400$.
        # Incremental is ~10x faster.
        
        # Let's implement incremental update for SOS.
        # R = ifft(|fft(h)|^2).real
        # We maintain R array.
        
        h = np.random.choice([-1.0, 1.0], N)
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H) ** 2).real
        current_sos = np.sum(R[1:]**2)
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 1e-4
        steps_per_restart = 100000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            idx = np.random.randint(0, N)
            
            # Calculate delta SOS
            # R[k] changes by -2 * h[idx] * (h[(idx-k)%N] + h[(idx+k)%N])
            # We need to update R and then delta_sos.
            # delta_sos = sum_{k=1}^{N-1} (R[k] + delta_R[k])^2 - R[k]^2
            # This is still O(N) to compute if we do it naively.
            # But we can just recompute R from scratch if FFT is fast enough?
            # Let's benchmark mentally.
            # Python FFT overhead is high. Incremental update in Python loop is also high.
            # Vectorized incremental update:
            # delta_R = -2 * h[idx] * (np.roll(h, -idx) + np.roll(h, idx))
            # R_new = R + delta_R
            # This is 2 rolls and some arithmetic. O(N).
            
            delta_R = -2.0 * h[idx] * (np.roll(h, -idx) + np.roll(h, idx))
            # R[0] is energy, it doesn't change for +/- 1 flips?
            # sum h[i]^2 is constant. So R[0] is constant.
            # We only care about R[1:].
            
            # Update R
            R += delta_R
            
            new_sos = np.sum(R[1:]**2)
            delta_sos = new_sos - current_sos
            
            if delta_sos < 0 or np.random.random() < np.exp(-delta_sos / T):
                h[idx] *= -1
                current_sos = new_sos
            else:
                R -= delta_R # Revert
            
            T *= cooling_rate
            
        # Phase 2: Local Search on Max Abs
        # Convert to max metric search
        h_current = h.copy()
        current_max = _compute_max_metric(h_current)
        
        # Simple local search: flip bits that reduce max metric
        for ls_step in range(5000):
            if time.time() > deadline - 2:
                break
            
            idx = np.random.randint(0, N)
            h_trial = h_current.copy()
            h_trial[idx] *= -1
            
            trial_max = _compute_max_metric(h_trial)
            
            if trial_max < current_max:
                h_current = h_trial
                current_max = trial_max
                
                if current_max < best_max_score:
                    best_max_score = current_max
                    best_h = h_current.copy()
            else:
                # Optionally accept small increases to escape local minima in max metric?
                # Or just stop?
                pass

    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
