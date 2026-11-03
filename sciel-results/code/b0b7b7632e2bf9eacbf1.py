"""Spectral Flattening Strategy: Minimize off-DC spectral energy to suppress autoconvolution lobes, then refine peak."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Precompute indices for real FFT
_M = _FFT_SIZE // 2 + 1

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _soft_max_abs(conv, beta=50.0):
    """Smooth approximation of max(|conv|)."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _spectral_l2_metric(x, beta=50.0):
    """
    Metric to optimize:
    1. Primary: Soft max of autoconvolution (targets the peak).
    2. Secondary: L2 norm of non-DC spectral components (targets overall spread/lobes).
    
    We normalize by integral squared.
    """
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    
    conv = _autoconv_values(x)
    peak_soft = _soft_max_abs(conv, beta)
    
    # Spectral flattening term:
    # Minimize sum_{k != 0} |FX[k]|^2. 
    # By Parseval, this is proportional to sum x^2 - (sum x)^2 / N_fft.
    # Minimizing this encourages the signal to be as "flat" in time as possible 
    # (which corresponds to a delta in frequency, i.e., flat time signal? No, 
    # flat time signal is delta in frequency. 
    # Wait:
    # If x is constant (flat time), FX[0] is large, others 0. 
    # Then conv is triangular. 
    # If x is random (white noise), FX is flat. 
    # conv of white noise is spike at 0, small elsewhere.
    # So minimizing L2 of non-DC spectrum makes x "whiter" in time, 
    # which makes conv spikier (more like delta).
    # A spiky conv has a large peak at 0, but small peaks elsewhere.
    # The metric is max_abs(conv) / s^2.
    # If conv is a spike, max_abs is conv[0] = sum x^2.
    # s^2 = (sum x)^2.
    # For zero-mean-ish signals, this ratio can be small if sum x is large relative to energy?
    # No, s^2 <= N * sum x^2. So ratio >= 1/N.
    # Actually, we want to minimize max_abs(conv).
    # If we make conv a delta, max_abs is large (concentrated energy).
    # If we spread conv out, max_abs is smaller.
    # So we want to MAXIMIZE the L2 norm of non-DC spectrum? 
    # Or rather, we want the energy of conv to be spread out.
    # Energy of conv is fixed by x (Parseval: sum conv^2 = 1/F * sum |FX|^4).
    # To minimize max, we want conv to be as flat as possible.
    # Flat conv in time <-> Flat spectrum of conv in frequency.
    # Spectrum of conv is |FX|^2.
    # So we want |FX|^2 to be flat!
    # This means |FX[k]| should be constant for all k.
    # This is the "Flat Spectrum" design.
    
    # So the objective should penalize deviation of |FX[k]| from the mean.
    # Variance of |FX|^2.
    
    fx = np.fft.rfft(x, _FFT_SIZE)
    mags_sq = np.abs(fx)**2
    # Exclude DC (k=0) and Nyquist (k=M-1 if even) from variance? 
    # DC is fixed by s. 
    # Let's look at k=1 to M-1.
    if _M > 2:
        non_dc_mags = mags_sq[1:_M]
        # If FFT size is even, last bin is real, but still part of spectrum.
        # Variance of non-DC magnitudes
        var = np.var(non_dc_mags)
    else:
        var = 0.0
        
    # Combine: We want low peak (soft_max) and flat spectrum (low variance of mags).
    # Normalize by s^2 to handle scale invariance.
    # Note: soft_max is O(1) relative to x. var is O(x^4).
    # Let's use a weighted sum.
    
    return (peak_soft + 0.01 * var) / (s * s)

def _try_spectral_flattening(x_init, deadline, rng):
    """
    Optimize the signal in the frequency domain to flatten the magnitude spectrum.
    We parameterize x via its FFT coefficients.
    We fix FX[0] = s (integral).
    We optimize the phases and magnitudes of FX[k] for k>0.
    
    Actually, optimizing magnitudes and phases jointly is non-convex.
    However, the "flat spectrum" solution is known: 
    If we can make |FX[k]| constant, we get a signal whose autoconvolution is 
    as flat as possible (in a least-squares sense).
    
    Let's use scipy.optimize.minimize on the time-domain signal x, 
    but with the spectral flattening objective.
    """
    x = x_init.copy()
    s = _integral(x)
    if abs(s) < 1e-12:
        return x, _metric(x)
    
    # Normalize to s=1 for optimization stability, then scale back if needed?
    # The metric is scale invariant: metric(cx) = metric(x).
    # So we can just fix sum(x)=1 during optimization.
    
    x = x / s
    
    def obj(v):
        # v is the vector x
        # Constraint: sum(v) = 1 is handled by projection or penalty?
        # Let's use a penalty for deviation from sum=1.
        # Or better, optimize v[1:] and set v[0] = 1 - sum(v[1:])?
        # No, that biases index 0.
        # Let's just use the metric which is scale invariant, 
        # but add a large penalty if sum is far from 1 to keep it well-conditioned.
        s_v = np.sum(v)
        if abs(s_v) < 1e-6:
            return 1e10
        penalty = 100.0 * (s_v - 1.0)**2
        
        return _spectral_l2_metric(v, beta=50.0) + penalty

    def grad(v):
        # Numerical gradient for simplicity, or analytic?
        # N=400, numerical gradient takes 400 evaluations. 
        # Each eval is O(N log N). 400 * 400 * 9 ~ 1.4M ops. Fast enough.
        eps = 1e-6
        g = np.zeros(N)
        f0 = obj(v)
        for i in range(N):
            v_plus = v.copy()
            v_plus[i] += eps
            g[i] = (obj(v_plus) - f0) / eps
        return g

    # Use L-BFGS-B
    options = {'maxiter': 200, 'disp': False}
    
    # Multiple restarts with different initial phases/magnitudes
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Restart 1: Start from current x
    try:
        res = minimize(obj, x, method='L-BFGS-B', jac=grad, options=options)
        if res.success and _metric(res.x) < best_val:
            best_val = _metric(res.x)
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Restart 2: Random phases, flat magnitudes
    # Flat magnitudes: |FX[k]| = C for all k.
    # sum x = FX[0] = 1.
    # Energy: sum x^2 = 1/F * sum |FX|^2 = 1/F * (1 + (M-1)C^2).
    # We don't care about energy directly, but flat spectrum is the target.
    # Let's try to initialize with random phases and equal magnitudes.
    
    fx = np.zeros(_M, dtype=complex)
    fx[0] = 1.0
    # Choose C such that x is roughly O(1)?
    # If C=1, sum |FX|^2 = 1 + M. sum x^2 = (1+M)/F.
    # F ~ 2N. So sum x^2 ~ 1. x is O(1/sqrt(N))? No, x ~ 1/sqrt(N) if energy 1.
    # Let's just pick C=1.
    fx[1:] = np.exp(1j * rng.uniform(0, 2*np.pi, size=_M-1))
    x_rand = np.fft.irfft(fx, _FFT_SIZE)[:N]
    
    # Normalize x_rand to sum=1
    s_rand = np.sum(x_rand)
    if abs(s_rand) > 1e-6:
        x_rand = x_rand / s_rand
        try:
            res2 = minimize(obj, x_rand, method='L-BFGS-B', jac=grad, options=options)
            if res2.success and _metric(res2.x) < best_val:
                best_val = _metric(res2.x)
                best_x = res2.x.copy()
        except Exception:
            pass
            
    return best_x, best_val

def _try_peak_refinement(x_init, deadline, rng):
    """
    Refine the peak specifically using soft-max gradient descent.
    """
    x = x_init.copy()
    s = _integral(x)
    if abs(s) < 1e-12:
        return x, _metric(x)
    x = x / s
    
    def obj_peak(v):
        s_v = np.sum(v)
        if abs(s_v) < 1e-6:
            return 1e10
        penalty = 100.0 * (s_v - 1.0)**2
        conv = _autoconv_values(v)
        return _soft_max_abs(conv, beta=100.0) + penalty

    def grad_peak(v):
        eps = 1e-6
        g = np.zeros(N)
        f0 = obj_peak(v)
        for i in range(N):
            v_plus = v.copy()
            v_plus[i] += eps
            g[i] = (obj_peak(v_plus) - f0) / eps
        return g

    best_x = x.copy()
    best_val = _metric(best_x)
    
    options = {'maxiter': 100, 'disp': False}
    try:
        res = minimize(obj_peak, x, method='L-BFGS-B', jac=grad_peak, options=options)
        if res.success and _metric(res.x) < best_val:
            best_val = _metric(res.x)
            best_x = res.x.copy()
    except Exception:
        pass
        
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    # Initial guess: Random white noise (flat spectrum)
    best_x = rng.normal(size=N)
    best_val = _metric(best_x)
    
    # Also try constant signal
    x_const = np.ones(N)
    v_const = _metric(x_const)
    if v_const < best_val:
        best_val = v_const
        best_x = x_const.copy()
        
    # Phase 1: Spectral Flattening
    if time.time() < deadline - 30:
        x_flat, val_flat = _try_spectral_flattening(best_x, deadline - 30, rng)
        if val_flat < best_val:
            best_val = val_flat
            best_x = x_flat
    
    # Phase 2: Peak Refinement
    if time.time() < deadline - 5:
        x_ref, val_ref = _try_peak_refinement(best_x, deadline - 5, rng)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref
            
    return best_x
# EVOLVE-BLOCK-END
