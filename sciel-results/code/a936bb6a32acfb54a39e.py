import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    # signed_step is an array of N values in {-1, 1}
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _soft_max_abs(x, beta=50.0):
    """
    Smooth approximation of max(|x|) using log-sum-exp.
    """
    ax = np.abs(x)
    return np.log(np.sum(np.exp(beta * ax))) / beta

def _soft_c3_continuous(x, beta=50.0):
    """
    Compute soft C3 for continuous variables x in [-1, 1].
    x represents the heights of the step function.
    """
    M = 4 * N
    dx = (0.5) / M
    
    # Build f on fine grid using continuous values
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = x[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Soft max of |autoconv|
    c3 = _soft_max_abs(autoconv, beta)
    
    return c3

def _soft_c3_grad(x, beta=50.0):
    """
    Compute gradient of soft C3 with respect to x.
    """
    M = 4 * N
    dx = (0.5) / M
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = x[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Gradient of soft max: softmax-like weights
    ax = np.abs(autoconv)
    weights = np.exp(beta * ax) / np.sum(np.exp(beta * ax))
    
    # d/dx of autoconv: autoconv is quadratic in f, so gradient involves f
    # autoconv[t] = sum_s f[s] f[t-s] * dx
    # d(autoconv[t])/d(f[i]) = sum_s (delta_{s,i} f[t-s] + f[s] delta_{t-s,i}) * dx
    # = (f[t-i] + f[i-t]) * dx (with appropriate indexing)
    
    # For efficiency, compute gradient via FFT
    # autoconv = ifft(|fft(f)|^2)
    # d(autoconv)/d(f) = 2 * ifft(fft(f) * conj(fft(f))) ... this is circular
    # Actually, autoconv is the autocorrelation, so:
    # autoconv[t] = sum_s f[s] f[s+t]
    # d(autoconv[t])/d(f[i]) = f[i+t] + f[i-t] (with wraparound considerations)
    
    # Simpler: use the fact that autoconv is a quadratic form
    # autoconv = (f * f)(t) where * is convolution
    # In frequency domain: autoconv = ifft(|F|^2)
    # d(autoconv)/d(f) can be computed as 2 * ifft(F * conj(F) * ... )
    # Actually, let's just compute it numerically or use the structure
    
    # For a step function, the autoconvolution at lag k is:
    # R[k] = sum_{i=0}^{N-k-1} f[i] f[i+k] * (interval_width)
    # So dR[k]/df[j] = f[j+k] + f[j-k] (with bounds checking)
    
    # Let's compute the gradient of the soft max w.r.t. autoconv first
    d_autoconv = weights * np.sign(autoconv)
    
    # Now compute d(autoconv)/d(f) for each interval
    grad_x = np.zeros(N)
    interval_width = 0.5 / N
    
    # For each lag k, R[k] = sum_i f[i] f[i+k] * interval_width
    # dR[k]/df[j] = f[j+k] * interval_width + f[j-k] * interval_weight (with bounds)
    
    for k in range(-N+1, N):
        if k >= 0:
            # R[k] = sum_{i=0}^{N-k-1} f[i] f[i+k] * interval_width
            for j in range(N - k):
                # dR[k]/df[j] = f[j+k] * interval_width
                # dR[k]/df[j+k] = f[j] * interval_width
                grad_x[j] += d_autoconv[k + N - 1] * f_fine[j + k] * interval_width
                grad_x[j + k] += d_autoconv[k + N - 1] * f_fine[j] * interval_width
        else:
            # R[k] = R[-k] by symmetry for real signals, but let's handle it
            kk = -k
            for j in range(N - kk):
                grad_x[j] += d_autoconv[k + N - 1] * f_fine[j + kk] * interval_width
                grad_x[j + kk] += d_autoconv[k + N - 1] * f_fine[j] * interval_width
    
    return grad_x

def _continuous_optimize(x0, beta=50.0, max_iter=100, time_limit=30.0):
    """
    Run L-BFGS-B optimization on continuous variables.
    """
    start_time = time.time()
    
    def objective(x):
        return _soft_c3_continuous(x, beta)
    
    def gradient(x):
        return _soft_c3_grad(x, beta)
    
    bounds = [(-1.0, 1.0)] * N
    
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B', 
        jac=gradient,
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    # Check time
    elapsed = time.time() - start_time
    
    return result.x, result.fun, elapsed

def _discrete_from_continuous(x):
    """Convert continuous values to discrete ±1."""
    return np.where(x >= 0, 1, -1).astype(np.int8)

def construct():
    start_time = time.time()
    
    np.random.seed(42)
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Phase 1: Continuous optimization with tanh relaxation
    # Try multiple initializations
    initializations = [
        np.ones(N),  # all positive
        np.ones(N) * -1,  # all negative
        np.random.choice([-1, 1], size=N).astype(float),  # random
        np.tile([-1, 1], N // 2).astype(float),  # alternating
        np.tile([1, 1, -1, -1], N // 4).astype(float),  # block pattern
    ]
    
    for x0 in initializations:
        if time.time() - start_time > DEADLINE - 40.0:
            break
        
        try:
            x_opt, fun_val, elapsed = _continuous_optimize(x0, beta=50.0, max_iter=50, time_limit=20.0)
            
            # Convert to discrete
            step_candidate = _discrete_from_continuous(x_opt)
            c3_candidate = _compute_c3(step_candidate)
            
            if c3_candidate < best_c3:
                best_c3 = c3_candidate
                best_step = step_candidate.copy()
        except Exception:
            continue
    
    # Phase 2: Discrete hill-climbing refinement
    no_improve_count = 0
    patience = 3000
    
    while time.time() - start_time < DEADLINE - 2.0:
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            best_step[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count >= patience:
            break
    
    return best_step
