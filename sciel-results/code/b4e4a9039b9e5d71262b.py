import numpy as np
import time

N = 400

def _autocorr_peak(s):
    """Compute peak sidelobe of linear autocorrelation for signed sequence s."""
    # Linear autocorrelation
    n = len(s)
    ac = np.correlate(s, s, mode='full')
    # Peak is at center; sidelobes are the rest
    # ac has length 2n-1, center at n-1
    center = n - 1
    peak = ac[center]
    # Peak sidelobe: max absolute value away from center
    if n > 1:
        sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        peak_side = np.max(np.abs(sidelobes))
    else:
        peak_side = 0.0
    return peak_side

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s (and 150s kill)
    
    # Gumbel-Softmax based gradient descent
    # Parameters: logits for each of N bits
    np.random.seed(42)
    logits = np.zeros(N)
    lr = 0.1
    temperature = 1.0
    
    best_val = None
    best_s = np.ones(N)
    
    # Evaluate seed
    seed_val = _autocorr_peak(best_s)
    best_val = seed_val
    
    # Gradient descent with Gumbel-Softmax
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Sample from Gumbel-Softmax
        eps = 1e-9
        gumbel = -np.log(-np.log(np.random.rand(N) + eps) + eps)
        z = (logits + gumbel) / temperature
        # Soft sign: sigmoid-like
        s_soft = 2.0 * (1.0 / (1.0 + np.exp(-z))) - 1.0  # in (-1, 1)
        
        # Compute objective: peak sidelobe (smooth approximation via soft-max)
        # Use a smooth approximation of max via log-sum-exp
        ac = np.correlate(s_soft, s_soft, mode='full')
        center = N - 1
        sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        abs_sl = np.abs(sidelobes)
        
        # Smooth max via log-sum-exp
        tau = 0.5
        max_val = np.logaddexp.reduce(np.log(abs_sl + eps) / tau) * tau
        
        # Gradient of max_val w.r.t. s_soft
        # d/ds of logsumexp(log(|ac|)/tau) * tau
        # = (1/tau) * sum_i exp(log(|ac_i|)/tau) * (1/|ac_i|) * d|ac_i|/ds * tau
        # = sum_i (|ac_i|/Z) * sign(ac_i) * dac_i/ds
        # where Z = sum |ac_i|
        weights = abs_sl / (np.sum(abs_sl) + eps)
        # Gradient of ac w.r.t. s_soft
        # ac[k] = sum_i s[i] s[i+k] (for appropriate indices)
        # d ac[k] / d s[j] = s[j+k] + s[j-k] (with bounds)
        # This is expensive; use finite difference or analytical
        # For efficiency, use the fact that autocorr gradient can be computed via convolution
        # dac/ds[j] = 2 * (s[j] convolved with s) at appropriate lags
        # Actually: ac = s * s (correlation), d(ac)/d(s[j]) = 2 * (s shifted)
        # Let's compute: ac_full[m] = sum_i s[i] s[i + m - (N-1)]
        # d ac_full[m] / d s[j] = s[j + m - (N-1)] + s[j - (m - (N-1))] if in bounds
        
        # Simpler: use the property that d/ds of autocorr is 2 * (s correlated with itself shifted)
        # For a given j, d ac / d s[j] has contributions at all lags
        # Let's just use a numerical gradient for the key terms or use the structure
        
        # Actually, let's use a different approach: since N=400, computing the full gradient
        # via the correlation structure:
        # ac_full[m] for m in [0, 2N-2]
        # ac_full[m] = sum_{i} s[i] * s[i + m - N + 1] where indices valid
        # d ac_full[m] / d s[j] = s[j + m - N + 1] (if valid) + s[j - (m - N + 1)] (if valid)
        
        # This is O(N^2) per step which is 160k operations - fine for 20k steps? 
        # 20k * 160k = 3.2e9 - too slow.
        
        # Alternative: use the fact that the gradient of the smooth max can be computed
        # and then backprop through the correlation using FFT
        # d ac / d s = 2 * (s * s) in some sense... 
        # Actually for autocorrelation: if y = x * x (convolution with reversed), 
        # then dy/dx = 2 * (x * x_reversed) which is just 2 * ac itself in a sense.
        
        # Let me use a simpler approach: coordinate descent or local search
        # given time constraints.
        
        # Let me switch to a local search / simulated annealing approach which is faster
        break
    
    # Use local search instead - much faster per iteration
    # Simulated annealing / local search
    s = np.ones(N)
    current_val = _autocorr_peak(s)
    best_val = current_val
    best_s = s.copy()
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Random flip
        i = np.random.randint(N)
        s[i] = -s[i]
        new_val = _autocorr_peak(s)
        delta = new_val - current_val
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-9)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[i] = -s[i]  # revert
        
        # Periodically check time
        if time.time() > deadline:
            break
    
    return best_s
