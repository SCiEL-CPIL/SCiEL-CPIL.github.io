import numpy as np
import time

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[0] is the DC term, R[1:] and R[-(N-1):] are non-zero lags
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _compute_gradient(h):
    """Compute approximate gradient of the metric w.r.t. h."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    
    # The gradient of max|R[k]| w.r.t. h can be approximated
    # For the peak lag, use the phase of the autocorrelation
    peak_lag = np.argmax(np.abs(R[1:])) + 1
    if peak_lag >= N:
        peak_lag -= N
    
    # Gradient of R[peak_lag] w.r.t. h is related to circular correlation
    # dR[k]/dh[i] = h[i+k] (circular)
    grad = np.roll(h, -peak_lag)
    
    # Scale by the sign of the peak to minimize its absolute value
    if R[peak_lag] > 0:
        grad *= 1.0
    else:
        grad *= -1.0
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Start with all ones
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = _compute_metric(best_h)
    
    # Simulated annealing parameters
    T_start = 0.1
    T_end = 0.001
    num_iterations = 20000
    
    h_current = best_h.copy()
    current_score = best_score
    
    for it in range(num_iterations):
        if time.time() > deadline - 3:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (it / num_iterations)
        
        # Generate a move: flip a random subset of coordinates
        num_flips = np.random.randint(1, 6)  # 1-5 coordinates
        flip_indices = np.random.choice(N, num_flips, replace=False)
        
        old_values = h_current[flip_indices].copy()
        
        # Try flipping signs or setting to 0
        for idx in flip_indices:
            # Randomly choose: flip sign, set to 0, or keep
            choice = np.random.random()
            if choice < 0.5:
                h_current[idx] = -h_current[idx]
            elif choice < 0.7:
                h_current[idx] = 0.0
            # else keep
            
        new_score = _compute_metric(h_current)
        
        # Accept if better or with probability exp(-(new_score - current_score) / T)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_h = h_current.copy()
        else:
            # Metropolis acceptance
            delta = new_score - current_score
            if delta < T and np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Reject: revert
                h_current[flip_indices] = old_values
    
    # Final polish: local search with tighter tolerance
    h_polish = best_h.copy()
    polish_score = best_score
    for polish_iter in range(500):
        if time.time() > deadline - 2:
            break
        
        improved = False
        # Try single-coordinate updates
        coords = np.random.choice(N, 20, replace=False)
        for idx in coords:
            if time.time() > deadline - 2:
                break
            
            old_val = h_polish[idx]
            candidates = [1.0, -1.0, 0.0]
            
            for cval in candidates:
                if cval == old_val:
                    continue
                h_polish[idx] = cval
                score = _compute_metric(h_polish)
                if score < polish_score - 1e-14:
                    polish_score = score
                    improved = True
                    break
                else:
                    h_polish[idx] = old_val
            
            if not improved and time.time() > deadline - 2:
                break
        
        if not improved:
            break
    
    if polish_score < best_score - 1e-12:
        best_score = polish_score
        best_h = h_polish.copy()
    
    return best_h
