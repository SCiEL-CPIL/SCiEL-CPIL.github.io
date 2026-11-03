"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _spectral_flat_design(x_init, deadline, rng):
    """
    Design x to have a flat magnitude spectrum by iteratively
    shrinking large spectral components.
    """
    n = len(x_init)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    x = x_init.copy()
    # Normalize
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Learning rate
    # The update is X <- X * (1 - alpha * |X|^2)
    # We need to be careful with stability.
    # Let's use a fixed point iteration or gradient descent with step size.
    
    # We want to minimize sum |X_k|^4 for k != 0.
    # This is equivalent to minimizing the peakiness of the autocorrelation tail.
    
    eta = 0.5 # Initial learning rate
    no_improve = 0
    
    for i in range(10000):
        if time.time() > deadline:
            break
        
        # Transform to frequency
        fx = np.fft.rfft(x, size)
        
        # Compute gradient of J = sum_{k!=0} |X_k|^4 / size
        # dJ/dX_k = 4/size * |X_k|^2 * X_k
        # We want to subtract this.
        # Note: DC component (k=0) should be kept fixed or handled carefully
        # because we normalize to sum=1.
        
        # Mask out DC for the gradient?
        # The constraint is sum(x) = 1, which is X[0] = 1 (if real).
        # So we should not update X[0].
        
        mask = np.ones(len(fx), dtype=bool)
        mask[0] = False
        
        grad = np.zeros_like(fx)
        mag_sq = np.abs(fx[mask])**2
        grad[mask] = (4.0 / size) * mag_sq * fx[mask]
        
        # Update
        x_new_f = fx.copy()
        x_new_f[mask] = fx[mask] - eta * grad[mask]
        
        # Transform back
        x_new = np.fft.irfft(x_new_f, size)
        x_new = x_new[:n] # Truncate to N
        
        # Zero out the tail if we didn't truncate correctly? 
        # irfft returns size/2+1 for real, but we passed size.
        # Wait, np.fft.irfft(fx, size) returns an array of length size.
        # So we take the first n samples.
        
        # Normalize to sum=1
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            # Failed normalization, break
            break
        x_new = x_new / s_new
        
        v_new = _metric(x_new)
        
        if v_new < best_val:
            best_val = v_new
            best_x = x_new.copy()
            no_improve = 0
            # Adaptive step size: if improving, maybe keep eta
        else:
            no_improve += 1
            if no_improve > 10:
                # Reduce eta
                eta *= 0.5
                if eta < 1e-4:
                    break
                no_improve = 0
        
    return best_x, best_val


def _local_search_peak_shaving(x, deadline, rng):
    """
    Local search: find the peak in autoconvolution, and try to reduce it
    by flipping or adjusting bits in x.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    
    n = len(best_x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
        
    # Precompute FFT of x
    fx = np.fft.rfft(best_x, size)
    
    no_improve = 0
    
    while time.time() < deadline:
        # Compute conv
        conv = np.fft.irfft(fx * fx, size)
        conv = conv[:2 * n - 1]
        
        # Find peak (excluding k=0? No, include all, but k=0 is usually largest)
        # The metric is max over all k.
        # Usually k=0 is the biggest. But if we flatten spectrum, k=0 stays, others drop.
        # We want to reduce max.
        
        peak_idx = np.argmax(np.abs(conv))
        
        # Try to reduce this peak.
        # Which x[i] affect conv[peak_idx]?
        # conv[k] = sum x[i] x[k-i].
        
        # Randomly pick a few indices to perturb
        k_perturb = rng.integers(1, 5)
        indices = rng.choice(n, size=k_perturb, replace=False)
        
        x_new = best_x.copy()
        
        # Try flipping signs or small shifts?
        # Since we are in continuous domain (step heights), we can do small steps.
        # But the problem says "signed step heights", implying discrete values?
        # "sequence of signed step heights".
        # The seed used -1, 1.
        # The inspirations used continuous values.
        # Let's stick to continuous optimization as it's more powerful.
        
        # Gradient step towards reducing conv[peak_idx]
        # This is complex. Let's use random restarts if stuck.
        
        # Simple random walk
        deltas = rng.normal(0, 0.01, k_perturb)
        x_new[indices] += deltas
        
        v_new = _metric(x_new)
        
        if v_new < best_val:
            best_val = v_new
            best_x = x_new
            fx = np.fft.rfft(best_x, size)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 500:
                break

    return best_x, best_val


def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 140.0

    rng = np.random.default_rng(2024)

    # Phase 1: Spectral Flat Design
    # Start with random x
    x_init = rng.normal(0, 1, N)
    
    best_x, best_val = _spectral_flat_design(x_init, deadline - 10, rng)
    
    # Phase 2: Local Search / Polish
    best_x, best_val = _local_search_peak_shaving(best_x, deadline, rng)
    
    return best_x
# EVOLVE-BLOCK-END
