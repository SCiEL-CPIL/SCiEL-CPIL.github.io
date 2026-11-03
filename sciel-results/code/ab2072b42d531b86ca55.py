import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq

N = 600
TIME_LIMIT = 140  # Safety margin for 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta=10.0):
    """
    Smooth approximation of max(x) using log-sum-exp.
    """
    max_x = np.max(x)
    x_scaled = x - max_x
    return max_x + np.log(np.sum(np.exp(beta * x_scaled))) / beta

def phase_to_signal(phases, mags):
    """
    Construct the time-domain signal from magnitude and phase spectra.
    f(t) = | ifft( mags * exp(1j * phases) ) |
    We ensure the signal is real and non-negative by taking the absolute value.
    Note: For the inverse FFT to be real, the spectrum must be Hermitian symmetric.
    We handle this by defining phases for the positive frequencies and mirroring them.
    """
    M = len(phases)
    # mags is the magnitude spectrum for the first M+1 bins (0 to M)
    # We need to construct the full spectrum of length N
    full_mags = np.zeros(N)
    full_phases = np.zeros(N)
    
    # DC component (index 0)
    full_mags[0] = mags[0]
    full_phases[0] = 0.0
    
    # Positive frequencies (1 to M)
    # We assume M < N/2
    full_mags[1:M+1] = mags[1:M+1]
    full_phases[1:M+1] = phases
    
    # Negative frequencies (Hermitian symmetry)
    # Index N-k corresponds to -k
    for k in range(1, M+1):
        full_mags[N-k] = mags[k]
        full_phases[N-k] = -phases[k]
        
    # If N is even, the Nyquist frequency (N/2) must be real
    if N % 2 == 0:
        nyq_idx = N // 2
        if nyq_idx <= M:
            full_mags[nyq_idx] = mags[nyq_idx]
            full_phases[nyq_idx] = 0.0 # Phase must be 0 or pi for real signal, let's force 0 for simplicity
            
    spectrum = full_mags * np.exp(1j * full_phases)
    
    # Inverse FFT
    f_raw = np.real(ifft(spectrum))
    
    # Enforce non-negativity
    f = np.abs(f_raw)
    
    # Normalize to sum 1
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    else:
        f = np.ones(N) / N
        
    return f

def objective_and_grad(params, mags, beta=50.0):
    """
    Objective: Smooth max of autoconvolution.
    params: phase angles.
    """
    f = phase_to_signal(params, mags)
    conv = np.convolve(f, f, mode='full')
    
    # We want to minimize the max of the autoconvolution.
    # The autoconvolution peak is related to the L2 norm of the signal in frequency domain?
    # Actually, max(conv) is the peak of |F|^2 in frequency domain? No.
    # conv = f * f.
    # Let's just use smooth max of the time-domain conv.
    val = smooth_max(conv, beta)
    
    # Gradient calculation is complex here because of the abs() and normalization.
    # We will rely on numerical gradients via L-BFGS-B.
    return val

def run_phase_optimization(time_budget):
    """
    Run L-BFGS-B optimization on phase angles.
    """
    start_time = time.time()
    
    # Number of phase parameters to optimize.
    # We use the first M frequency bins.
    M = 100 # 100 degrees of freedom for phases
    
    # Initial phases: Random
    np.random.seed(42)
    phases0 = np.random.uniform(-np.pi, np.pi, M)
    
    # Magnitude spectrum:
    # A good magnitude spectrum for minimizing peak-to-average is often flat (white noise)
    # or decaying slowly. However, we want a "bell" shape in time, which implies a broad spectrum.
    # Let's try a flat magnitude spectrum first, then optimize phases to create a specific shape.
    # Actually, if we fix mags to be constant, the time signal is a sum of cosines with random phases.
    # The average value will be 0, but abs() makes it positive.
    # Let's use a decaying magnitude to ensure convergence and a specific shape.
    # mags[k] = 1 / (1 + k^alpha)
    mags = np.zeros(M+1)
    mags[0] = 1.0 # DC offset to ensure non-negativity before abs
    for k in range(1, M+1):
        mags[k] = 1.0 / (1 + k**0.5)
        
    # Normalize mags
    mags /= np.sum(mags)
    
    # Bounds for phases
    bounds = [(-np.pi, np.pi)] * M
    
    best_f = None
    best_score = np.inf
    
    try:
        # Use L-BFGS-B
        res = minimize(
            objective_and_grad,
            phases0,
            args=(mags,),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 300,
                'ftol': 1e-8,
                'gtol': 1e-5,
                'maxls': 50
            }
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_f = phase_to_signal(res.x, mags)
            
        # Also check the initial point
        f0 = phase_to_signal(phases0, mags)
        s0 = compute_score(f0)
        if s0 < best_score:
            best_score = s0
            best_f = f0
            
    except Exception as e:
        print(f"Phase optimization failed: {e}")
        
    if best_f is None:
        best_f = phase_to_signal(phases0, mags)
        best_score = compute_score(best_f)
        
    return best_f, best_score

def local_refinement(f0, time_budget):
    """
    Refine the time-domain function using a simple projected gradient descent.
    """
    f = f0.copy()
    start_time = time.time()
    best_score = compute_score(f)
    
    beta = 20.0
    step_size = 0.01
    
    for i in range(50):
        if time.time() - start_time > time_budget:
            break
            
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        
        # Weights for smooth max
        exp_terms = np.exp(beta * (conv - max_conv))
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of smooth_max(conv) w.r.t f
        # G[j] = 2 * sum_k weights[k] * (f_{k-j} + f_{j-k}) / 2 ?
        # d conv_k / d f_j = f_{k-j} + f_{j-k}
        # grad_f[j] = sum_k weights[k] * (f_{k-j} + f_{j-k})
        
        # Vectorized gradient computation using FFT
        # This is the cross-correlation of weights and f, plus its reverse.
        # Since f and weights are real, we can use FFT.
        # G = ifft(fft(weights) * fft(f)) ... this is convolution.
        # We need correlation.
        
        # Let's use a simple numerical gradient for the polish step to avoid bugs.
        # N=600 is small.
        eps = 1e-5
        grad_num = np.zeros(N)
        
        # To speed up, we only evaluate gradient for a subset of indices if time is tight,
        # but 600 evaluations of conv (O(N^2) or O(N log N)) is fast.
        # conv is O(N^2) = 360,000 ops. 600 * 360,000 = 216M ops.
        # This might be too slow in Python loops.
        # Let's use the analytical gradient via FFT.
        
        # grad_f = 2 * (weights * f) where * is convolution?
        # d (f*f) / d f = 2 * (f * delta) ? No.
        # Let h = weights.
        # J = h * f (convolution).
        # d J_k / d f_j = h_{k-j} + h_{j-k} ? No, h is fixed.
        # J_k = sum_i f_i h_{k-i}.
        # d J_k / d f_j = h_{k-j}.
        # So grad_f[j] = sum_k weights[k] h_{k-j} = sum_k weights[k] weights_{k-j} = (weights * weights)[j]?
        # Wait, the objective is smooth_max(conv).
        # conv = f * f.
        # Let C = conv.
        # d smooth_max(C) / d f_j = sum_k w_k d C_k / d f_j
        # d C_k / d f_j = f_{k-j} + f_{j-k}
        # So grad_f[j] = sum_k w_k (f_{k-j} + f_{j-k})
        
        # Term 1: sum_k w_k f_{k-j}. This is cross-correlation of w and f.
        # Term 2: sum_k w_k f_{j-k}. This is cross-correlation of w and f (reversed index).
        # Since f and w are real, Term 2 is the same as Term 1 if w is symmetric? No.
        # Term 1: (w cross-correlate f)[j]
        # Term 2: (w cross-correlate f)[j] ?
        # Let's check indices.
        # sum_k w_k f_{k-j}. Let m = k-j => k = m+j. sum_m w_{m+j} f_m.
        # This is correlation of w and f at lag j.
        # sum_k w_k f_{j-k}. Let m = j-k => k = j-m. sum_m w_{j-m} f_m.
        # This is also correlation of w and f at lag j?
        # w_{j-m} vs w_{m+j}.
        # If w is symmetric, they are the same.
        # conv is symmetric, so weights are symmetric.
        # So grad_f[j] = 2 * corr(w, f)[j].
        
        # Compute correlation via FFT
        # corr(w, f)[j] = ifft(fft(w) * conj(fft(f)))
        
        F_w = fft(weights)
        F_f = fft(f)
        corr = np.real(ifft(F_w * np.conj(F_f)))
        
        grad_f = 2.0 * corr
        
        # Projected Gradient Descent
        f_new = f - step_size * grad_f
        f_new = np.maximum(f_new, 0)
        s = np.sum(f_new)
        if s > 1e-15:
            f_new = f_new / s
        
        new_score = compute_score(f_new)
        if new_score < best_score:
            best_score = new_score
            f = f_new
            step_size *= 0.95
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return f, best_score

def construct():
    start_time = time.time()
    
    # 1. Phase Optimization
    time_budget_phase = 60.0
    
    f_phase, score_phase = run_phase_optimization(time_budget_phase)
    
    best_f = f_phase
    best_score = score_phase
    
    # 2. Local Refinement
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5.0
    
    if remaining_time > 5.0:
        f_refined, score_refined = local_refinement(best_f, remaining_time)
        
        if score_refined < best_score:
            best_f = f_refined
            best_score = score_refined
    
    # Final cleanup
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    # Ensure it's a list of floats
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
