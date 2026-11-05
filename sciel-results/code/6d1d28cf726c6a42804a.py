"""
Combined strategy: Spectral flattening (minimize L2 energy of autoconv) + soft-max surrogate + FFT autoconv + L-BFGS-B refinement.

Key insight from knowledge: "Spectral flattening via minimization of autocorrelation energy (L2 norm) reduces peak-to-average ratio more effectively than simple gradient descent on the max index, approaching the theoretical lower bound of ~1"

The approach:
1. Primary optimization: Minimize the L2 norm (energy) of the autoconvolution, which flattens the spectrum and reduces peak-to-average ratio.
2. Secondary refinement: Use soft-max surrogate to directly minimize the peak value.
3. Use FFT-based autoconvolution for speed.
4. Multi-phase optimization with increasing beta for soft-max.
5. Random restarts and coordinate descent polish.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = rfft(h, n=n)
    c = irfft(H * H, n=n)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    sum_exp = np.sum(np.exp(beta * x_shifted))
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _objective_energy(params, beta_energy=1.0):
    """
    Objective: Minimize L2 energy of autoconvolution (spectral flattening).
    This is the primary strategy that was found to be effective.
    """
    h = np.tanh(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    # L2 energy of autoconvolution, normalized by sum^2
    energy = np.sum(c * c) / (s * s)
    return energy

def _objective_soft_max(params, beta):
    """Soft-max version for gradient-based optimization of peak value."""
    h = np.tanh(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return abs(val)

def _objective_combined(params, beta, energy_weight=0.1):
    """Combined objective: peak value + energy_weight * L2 energy."""
    h = np.tanh(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    peak_val = 2.0 * N * sm / (s * s)
    energy = np.sum(c * c) / (s * s)
    return abs(peak_val) + energy_weight * energy

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # should be 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val
    
    # === Phase 0: Initialize with diverse starting points ===
    # Start 1: All ones (baseline)
    # Start 2: Random walk with zero mean
    # Start 3: Sine wave patterns
    # Start 4: Sparse spikes
    
    init_params_list = []
    
    # All ones: tanh(x) ≈ 1 when x is large positive
    init_params_list.append(np.full(N, 5.0))
    
    # Random walk
    for seed in range(5):
        rng_local = np.random.default_rng(seed)
        steps = rng_local.standard_normal(N) * 0.5
        h_rw = np.cumsum(steps)
        h_rw = h_rw - np.mean(h_rw) + 1.0  # shift to positive
        # Convert to params: params = atanh(h) but h must be in (-1,1)
        # Scale h_rw to be in (-0.9, 0.9)
        h_scaled = np.tanh(h_rw * 0.5)
        init_params_list.append(h_scaled * 5.0)  # scale up for tanh
    
    # Sine waves
    for freq in [1, 2, 5, 10]:
        x = np.linspace(0, 2 * np.pi * freq, N)
        h_sin = np.tanh(np.sin(x) * 0.5 + 1.0)
        init_params_list.append(h_sin * 5.0)
    
    # Sparse spikes: background + a few negative spikes
    for n_spikes in [3, 5, 10]:
        h_sparse = np.full(N, 1.0)
        positions = rng.choice(N, size=n_spikes, replace=False)
        h_sparse[positions] = -0.5
        init_params_list.append(np.tanh(h_sparse) * 5.0)
    
    # === Phase 1: Energy minimization (spectral flattening) ===
    # This is the key strategy that was found effective
    best_params_energy = init_params_list[0].copy()
    best_energy = _objective_energy(best_params_energy)
    
    for init_params in init_params_list:
        if time.time() > deadline - 80.0:
            break
        try:
            res = minimize(
                lambda p: _objective_energy(p),
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            h_opt = np.tanh(res.x)
            val_opt = _evaluate_exact(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
            if res.fun < best_energy:
                best_energy = res.fun
                best_params_energy = res.x.copy()
        except Exception:
            pass
    
    # === Phase 2: Soft-max peak minimization with increasing beta ===
    best_params_peak = best_params_energy.copy()
    best_peak_val = _evaluate_exact(np.tanh(best_params_peak))
    
    for beta in [10.0, 50.0, 200.0, 500.0, 1000.0]:
        if time.time() > deadline - 50.0:
            break
        try:
            res = minimize(
                lambda p: _objective_soft_max(p, beta),
                best_params_peak,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            h_opt = np.tanh(res.x)
            val_opt = _evaluate_exact(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
            if res.fun < best_peak_val:
                best_peak_val = res.fun
                best_params_peak = res.x.copy()
        except Exception:
            pass
    
    # === Phase 3: Combined objective (peak + energy) ===
    best_params_combined = best_params_peak.copy()
    best_combined_val = _evaluate_exact(np.tanh(best_params_combined))
    
    for beta in [50.0, 200.0, 500.0]:
        for energy_weight in [0.01, 0.05, 0.1, 0.5]:
            if time.time() > deadline - 30.0:
                break
            try:
                res = minimize(
                    lambda p: _objective_combined(p, beta, energy_weight),
                    best_params_combined,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
                )
                h_opt = np.tanh(res.x)
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                if res.fun < best_combined_val:
                    best_combined_val = res.fun
                    best_params_combined = res.x.copy()
            except Exception:
                pass
    
    # === Phase 4: Coordinate descent polish ===
    h = np.tanh(best_params_combined)
    improved = True
    while improved and (time.time() - t0) < deadline - 10:
        improved = False
        for i in range(N):
            if time.time() - t0 > deadline - 10:
                break
            old_val = _evaluate_exact(np.tanh(best_params_combined))
            for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                test_params = best_params_combined.copy()
                test_params[i] += delta
                test_val = _evaluate_exact(np.tanh(test_params))
                if test_val < old_val - 1e-10:
                    best_params_combined = test_params
                    old_val = test_val
                    improved = True
                    break
    
    # === Phase 5: Random restarts if time permits ===
    if time.time() - t0 < deadline - 15:
        for restart in range(10):
            if time.time() - t0 > deadline - 15:
                break
            # Random perturbation of best
            noise = rng.normal(0, 0.3, N)
            p_try = best_params_combined + noise
            try:
                res = minimize(
                    lambda p: _objective_soft_max(p, 200.0),
                    p_try,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
                )
                h_opt = np.tanh(res.x)
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    best_params_combined = res.x.copy()
            except Exception:
                pass
    
    h = np.tanh(best_params_combined)
    
    # Safety check: ensure finite and sum != 0
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-10:
        return np.ones(N)
    
    return h
# EVOLVE-BLOCK-END
