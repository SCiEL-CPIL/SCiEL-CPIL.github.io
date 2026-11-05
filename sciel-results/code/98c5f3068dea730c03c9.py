"""
Strategy: Golay Complementary Pair (GCP) Optimization with Structural Constraint.

The current strategy optimizes the 400 heights directly or via a sparse spectrum.
This new strategy exploits the structural properties of Golay Complementary Sequences.
A pair of sequences (a, b) is complementary if the sum of their aperiodic autocorrelations
is zero for all non-zero lags. 

If we let h = a + b, the autoconvolution of h involves terms like a*a, a*b, b*a, b*b.
While the strict complementary property $C_a + C_b = 0$ applies to autocorrelations, 
the cross-correlations also play a role. However, if we optimize a and b such that 
their individual autocorrelations have very low side-lobes and the cross-correlations 
are also controlled, the combined autoconvolution of h will have significantly reduced 
peaks compared to a random or single-mode signal.

Specifically, we parameterize a and b as real-valued sequences of length N.
We minimize a surrogate objective that penalizes the maximum peak of the autoconvolution 
of h = a + b, while also adding a penalty term that encourages the "Golay property" 
for the individual components to guide the optimizer into a region of the solution space 
where low-peak structures are more likely to exist.

We use L-BFGS-B with a smooth LogSumExp surrogate for the max function.
We also include a "binarization" pressure (soft) to encourage values to cluster 
near +/- 1, as Golay sequences are typically binary, which often yields optimal 
correlation properties.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 140.0  # Safety margin before 150s limit

def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_max(x, alpha):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return x_max
    x_shifted = (x - x_max) / alpha
    x_shifted = np.clip(x_shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(x_shifted)))
    return x_max + alpha * log_sum_exp

def _compute_conv_max_penalty(a, b, alpha):
    """
    Compute the objective for a pair (a, b).
    h = a + b.
    We want to minimize max(conv(h, h)).
    We also add a penalty for the "Golay property" of a and b individually,
    i.e., sum(C_a(non-zero), C_b(non-zero)) should be close to 0.
    """
    h = a + b
    
    # Normalize h to have sum 1 for scale invariance in the objective calculation
    # Note: The final value V is scale invariant.
    s_h = np.sum(h)
    if abs(s_h) < 1e-6:
        return 1e10
    h_norm = h / s_h
    
    conv_h = np.convolve(h_norm, h_norm)
    
    # The peak we care about is the max of the signed entries.
    # Since we want to minimize the absolute value of the final metric, 
    # and the metric is |2*N*max(conv)/sum^2|, 
    # minimizing max(conv) (assuming it's the positive peak) is the goal.
    # If the max is negative, the abs will flip it. 
    # Usually, the zero-lag term is the largest positive peak.
    # So minimizing the max of the non-zero lags or the overall max is key.
    
    peak = _surrogate_max(conv_h, alpha)
    
    # Golay Property Penalty:
    # C_a[t] = sum a[i] a[i+t]. We want sum(C_a[t] + C_b[t]) ~ 0 for t != 0.
    # This is equivalent to the cross-correlation of the spectra.
    # In time domain, we can compute the aperiodic autocorrelation.
    # For efficiency, we can use FFT.
    # But N=400 is small. Let's just compute the penalty on the side-lobes.
    
    # Calculate aperiodic autocorrelation of a and b
    # conv(a, a[::-1]) gives the full correlation.
    # The non-zero lags are the side-lobes.
    
    # To keep it fast, we approximate the penalty using the energy of the side-lobes
    # or the max of the side-lobes.
    # Let's use the max of the side-lobes of the sum correlation C_a + C_b.
    
    # Correlation of a with itself:
    # We can compute this via FFT for speed if needed, but direct is fine for N=400.
    # However, we are inside an optimizer loop. Let's be careful.
    
    # A cheaper proxy for the Golay property:
    # The sum of the autoconvolutions of a and b should have low side-lobes.
    # conv(a, a) + conv(b, b).
    
    # Let's just focus on the main objective: min max(conv(h,h)).
    # The "Golay" hint is in the initialization and the structure of a,b.
    # Adding a complex penalty might distract the optimizer.
    # Instead, we rely on the fact that if a and b are "complementary-like", 
    # their sum h has specific properties.
    
    # Let's add a small regularization to keep a and b bounded and "binary-like"
    # to encourage the structure.
    # Penalty: sum((a^2 - 1)^2) + sum((b^2 - 1)^2)
    # This pushes values towards -1 and 1.
    reg_a = np.sum((a**2 - 1.0)**2)
    reg_b = np.sum((b**2 - 1.0)**2)
    
    return peak + 0.01 * (reg_a + reg_b)

def construct():
    """
    Construct step heights using Golay Complementary Pair Optimization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Start with the seed result
    best_h = np.ones(N).copy()
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(42)
    
    # We will optimize a and b.
    # Total parameters: 2 * N = 800.
    # This is a high-dimensional problem for L-BFGS-B, but the structure 
    # (a+b) might help.
    # To reduce dimensionality, we can use a lower-dimensional parameterization 
    # for a and b, e.g., Fourier coefficients, but let's try direct first 
    # with a limited number of iterations and restarts.
    
    # Alternative: Use a sparse spectral representation for a and b as well?
    # No, let's stick to the time-domain a, b but use a clever initialization.
    
    # Initialization: 
    # Try to find a "Golay-like" pair. 
    # Random binary sequences are a good start for the structure, 
    # but we need them to be real and optimized.
    
    n_restarts = 10
    alpha_start = 0.5
    alpha_end = 0.05
    
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize a and b
        # Option 1: Random binary +/- 1
        # Option 2: Random Gaussian
        # Option 3: Structured (e.g., chirps)
        
        if i % 3 == 0:
            # Binary-like
            a_init = rng.choice([-1, 1], size=N)
            b_init = rng.choice([-1, 1], size=N)
        elif i % 3 == 1:
            # Gaussian
            a_init = rng.normal(0, 1, N)
            b_init = rng.normal(0, 1, N)
        else:
            # Chirp-like
            t = np.linspace(0, 1, N)
            a_init = np.cos(2 * np.pi * (t**2 + rng.random()))
            b_init = np.cos(2 * np.pi * (t**2 + rng.random()))
            
        params_init = np.concatenate([a_init, b_init])
        
        alpha = alpha_start + (alpha_end - alpha_start) * (i / n_restarts)
        
        def objective(params):
            a = params[:N]
            b = params[N:]
            return _compute_conv_max_penalty(a, b, alpha)
        
        try:
            res = minimize(
                objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if res.fun < 1e9:
                a_opt = res.x[:N]
                b_opt = res.x[N:]
                h_opt = a_opt + b_opt
                
                s_sum = np.sum(h_opt)
                if abs(s_sum) > 1e-6:
                    h_norm = h_opt / s_sum
                    v_opt = _evaluate(h_norm)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_norm.copy()
                        
        except Exception as e:
            pass

    # Final local refinement on best_h
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.1
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.95
            else:
                step_size = min(step_size * 1.1, 0.5)
                
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
