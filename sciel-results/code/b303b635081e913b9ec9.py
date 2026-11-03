"""Rewritten program: spectral-flatness gradient descent with discrete refinement."""
import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution via FFT."""
    n = len(x)
    x_padded = np.zeros(2 * n, dtype=np.float64)
    x_padded[:n] = x
    X = np.fft.fft(x_padded)
    C = np.fft.ifft(X * X.conj()).real
    return C

def _score(x):
    """Compute max|autoconv| / (sum x)^2."""
    S = np.sum(x)
    if S == 0:
        return np.inf
    C = _autoconv(x)
    return np.max(np.abs(C)) / (S * S)

def _spectral_flatness(x):
    """
    Spectral flatness objective: encourage flat |X(k)|^2 for k != 0.
    Returns (objective_value, gradient).
    """
    n = len(x)
    x_pad = np.zeros(2 * n, dtype=np.float64)
    x_pad[:n] = x
    X = np.fft.fft(x_pad)
    power = np.abs(X) ** 2
    # Exclude DC (k=0) and Nyquist if even
    mask = np.ones(2 * n, dtype=bool)
    mask[0] = False
    if 2 * n % 2 == 0:
        mask[n] = False
    # Also exclude the conjugate symmetric part to avoid double counting? 
    # Actually for real signals, power is symmetric. We'll use all non-DC bins.
    # Log-flatness: minimize variance of log(power) or maximize entropy.
    # Simpler: minimize sum of (power - mean)^2 / mean^2 for non-DC bins
    p = power[mask]
    if np.any(p < 1e-12):
        p = np.maximum(p, 1e-12)
    log_p = np.log(p)
    mean_log = np.mean(log_p)
    # Objective: variance of log-power
    obj = np.var(log_p)
    
    # Gradient w.r.t. x
    # This is complex. Let's use a simpler surrogate: minimize max |C(k)| for k != 0
    # But that's non-smooth. 
    # Alternative: use the autoconvolution directly and apply subgradient.
    # Let's stick with a different approach: 
    # Minimize sum_{k!=0} C(k)^2 / S^2, which is related to energy in off-peak autocorr.
    # This is smooth and differentiable.
    
    # Let's redefine: objective = sum_{k=1}^{n-1} C(k)^2 / S^2
    # Gradient of C(k) w.r.t. x[i]:
    # C(k) = sum_j x[j] x[j+k]
    # dC(k)/dx[i] = x[i+k] (if i+k < n) + x[i-k] (if i-k >= 0)
    
    # Let's compute gradient numerically or analytically.
    # Analytical is faster.
    
    return obj

def _energy_objective(x):
    """
    Objective: sum_{k=1}^{n-1} C(k)^2 / S^2
    This is smooth and we can compute gradient.
    """
    n = len(x)
    C = _autoconv(x)
    S = np.sum(x)
    if S == 0:
        return np.inf, np.zeros(n)
    # C is indexed such that C[n+k] corresponds to lag k
    # We want lags 1 to n-1
    off_peak = np.concatenate([C[n+1:], C[n-1:0:-1]])  # lags 1..n-1 and -1..-(n-1)
    # Actually C[n+k] for k=1..n-1 are positive lags
    # C[n-k] for k=1..n-1 are negative lags
    # Since autoconv is symmetric, C[k] = C[-k]
    # So we just need positive lags
    pos_lags = C[n+1:n+n]  # lags 1 to n-1
    energy = np.sum(pos_lags ** 2)
    obj = energy / (S * S)
    
    # Gradient: d(obj)/dx[i]
    # obj = (1/S^2) * sum_k C(k)^2
    # d(obj)/dx[i] = (1/S^2) * sum_k 2*C(k) * dC(k)/dx[i] - (2/S^3) * S * (1/S^2) * sum_k C(k)^2
    #              = (2/S^2) * sum_k C(k) * dC(k)/dx[i] - (2/S^3) * sum_k C(k)^2
    
    # dC(k)/dx[i] for lag k:
    # C(k) = sum_j x[j] x[j+k]
    # dC(k)/dx[i] = x[i+k] (if 0 <= i+k < n) + x[i-k] (if 0 <= i-k < n)
    
    grad_C = np.zeros(n, dtype=np.float64)
    for k in range(1, n):
        ck = C[n+k]
        # Contribution to dC(k)/dx[i]
        # For i in [0, n-k-1]: dC(k)/dx[i] includes x[i+k]
        # For i in [k, n-1]: dC(k)/dx[i] includes x[i-k]
        if n - k > 0:
            grad_C[:n-k] += ck * x[k:]
        if k < n:
            grad_C[k:] += ck * x[:n-k]
    
    sum_C_sq = np.sum(pos_lags ** 2)
    grad = (2.0 / (S * S)) * grad_C - (2.0 / (S ** 3)) * sum_C_sq
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    # Start with random initialization
    best_x = np.random.choice([1, -1], size=N, p=[0.5, 0.5]).astype(np.float64)
    best_score = _score(best_x)
    
    # Gradient descent on continuous relaxation, then round
    x = best_x.copy()
    
    # Phase 1: Continuous gradient descent
    lr = 0.1
    for step in range(2000):
        if time.time() > deadline:
            break
        
        obj, grad = _energy_objective(x)
        if not np.isfinite(obj):
            # Reset if sum is zero
            x = np.random.choice([1, -1], size=N, p=[0.5, 0.5]).astype(np.float64)
            continue
        
        x = x - lr * grad
        # Project to [-1, 1] box
        x = np.clip(x, -1.0, 1.0)
        
        # Occasionally round and check
        if step % 50 == 0:
            x_rounded = np.sign(x)
            x_rounded[x_rounded == 0] = 1
            s = _score(x_rounded)
            if s < best_score:
                best_score = s
                best_x = x_rounded.copy()
    
    # Final rounding
    x_rounded = np.sign(x)
    x_rounded[x_rounded == 0] = 1
    s = _score(x_rounded)
    if s < best_score:
        best_score = s
        best_x = x_rounded.copy()
    
    # Phase 2: Local search refinement
    x = best_x.astype(np.float64)
    current_score = best_score
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Simulated annealing on discrete variables
        T = 0.01
        T_min = 1e-6
        cooling = 0.995
        
        for step in range(5000):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            idx = np.random.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            
            new_score = _score(x)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-12)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[idx] = old_val
            
            T *= cooling
        
        # Restart with perturbation
        for _ in range(10):
            idx = np.random.randint(N)
            x[idx] = -x[idx]
        current_score = _score(x)
    
    # Return as integer-like (but the contract says "sequence of signed step heights")
    # The seed program returned np.int8, so we do the same
    return best_x.astype(np.int8)
