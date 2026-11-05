"""Combined strategy: random-walk exploration + sparse-spike + full-vector L-BFGS-B with soft-max + coordinate descent polish."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400

# Precompute FFT size for autoconvolution
_FFT_N = 2 * N - 1

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    H = rfft(h, n=_FFT_N)
    c = irfft(H * H, n=_FFT_N)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    exp_vals = np.exp(beta * x_shifted)
    sum_exp = np.sum(exp_vals)
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

def _sparse_to_h(params, n_spikes=20):
    """
    Convert sparse parameterization to full height vector.
    params layout: [background (1), spike_amplitudes (n_spikes), spike_positions (n_spikes)]
    Spike positions are in [0, N), mapped to nearest integer index.
    Multiple spikes at same index sum their amplitudes.
    """
    bg = params[0]
    n = n_spikes
    amps = params[1:1+n]
    pos = params[1+n:1+2*n]
    
    h = np.full(N, bg)
    idx = np.round(pos).astype(int)
    idx = np.clip(idx, 0, N-1)
    np.add.at(h, idx, amps)
    return h

def _objective_sparse(params, beta, n_spikes=20):
    """Soft-max objective for sparse parameterization."""
    h = _sparse_to_h(params, n_spikes)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _objective_full(params, beta):
    """Soft-max objective for full h vector."""
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _random_walk_to_h(drift, noise, seed=42):
    """
    Generate h from a random walk with negative drift.
    h[i] = sum_{j=0}^{i} (noise[j] - drift)
    Then center to have zero mean, then add small positive offset.
    """
    rng_local = np.random.default_rng(seed)
    steps = rng_local.standard_normal(N) * noise - drift
    h = np.cumsum(steps)
    h = h - np.mean(h)
    h = h + 0.01
    return h

def _drift_objective(params, beta):
    """
    Objective for drift parameterization.
    params = [drift, noise_scale]
    """
    drift = params[0]
    noise = params[1]
    h = _random_walk_to_h(drift, noise)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin: evaluator kills at 150s

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

    # === Phase 0: Random walk exploration with drift optimization ===
    for seed in range(20):
        if time.time() > deadline - 55.0:
            break
        for drift in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for noise in [0.1, 0.5, 1.0, 2.0]:
                if time.time() > deadline - 55.0:
                    break
                h_rw = _random_walk_to_h(drift, noise, seed=seed)
                try_h(h_rw)

    # === Phase 1: Optimize drift/noise parameters with L-BFGS-B ===
    for beta in [10.0, 50.0, 200.0, 500.0]:
        if time.time() > deadline - 45.0:
            break
        for init_drift in [0.0, 0.05, 0.1, 0.2]:
            for init_noise in [0.5, 1.0, 2.0]:
                if time.time() > deadline - 45.0:
                    break
                init_params = np.array([init_drift, init_noise])
                try:
                    res = minimize(
                        lambda p: _drift_objective(p, beta),
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    h_opt = _random_walk_to_h(res.x[0], res.x[1])
                    val_opt = _evaluate_exact(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    # === Phase 2: Sparse-spike grid search ===
    n_spikes = 15

    bg_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    spike_amp_fracs = [0.5, 1.0, 2.0, 3.0, 5.0]
    freqs = [1, 2, 3, 5, 7, 10]

    for bg in bg_values:
        for freq in freqs:
            for amp_frac in spike_amp_fracs:
                if time.time() > deadline - 35.0:
                    break

                spike_positions = np.array([int((i * N) / freq) % N for i in range(freq)])
                spike_amps = np.full(freq, bg * amp_frac)

                h = np.full(N, bg)
                np.add.at(h, spike_positions, spike_amps)
                try_h(h)

                spike_amps_neg = np.full(freq, -bg * amp_frac)
                h_neg = np.full(N, bg)
                np.add.at(h_neg, spike_positions, spike_amps_neg)
                try_h(h_neg)

    # === Phase 3: Optimize sparse parameterization with L-BFGS-B ===
    bg_est = np.mean(best_h)
    devs = best_h - bg_est
    top_idx = np.argsort(np.abs(devs))[-n_spikes:]

    init_params = np.zeros(1 + 2 * n_spikes)
    init_params[0] = bg_est
    for i, idx in enumerate(top_idx):
        init_params[1 + i] = devs[idx]
        init_params[1 + n_spikes + i] = float(idx)

    for beta in [20.0, 100.0, 500.0, 1000.0]:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                lambda p: _objective_sparse(p, beta, n_spikes),
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = _sparse_to_h(res.x, n_spikes)
            val_opt = _evaluate_exact(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                bg_est = np.mean(best_h)
                devs = best_h - bg_est
                top_idx = np.argsort(np.abs(devs))[-n_spikes:]
                init_params = np.zeros(1 + 2 * n_spikes)
                init_params[0] = bg_est
                for i, idx in enumerate(top_idx):
                    init_params[1 + i] = devs[idx]
                    init_params[1 + n_spikes + i] = float(idx)
        except Exception:
            pass

    # === Phase 4: Full-vector L-BFGS-B refinement with soft-max ===
    for beta in [100.0, 500.0, 1000.0, 2000.0]:
        if time.time() > deadline - 20.0:
            break

        for restart in range(6):
            if time.time() > deadline - 20.0:
                break

            if restart == 0:
                x0 = best_h.copy()
            elif restart == 1:
                noise_scale = 0.02 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 2:
                noise_scale = 0.05 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 3:
                noise_scale = 0.10 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 4:
                mask = rng.random(N) < 0.3
                x0 = best_h.copy()
                x0[mask] += rng.standard_normal(np.sum(mask)) * 0.05 * np.std(best_h)
            else:
                x0 = np.ones(N)
                x0[rng.integers(0, N, size=N//4)] += 1.0
                x0[rng.integers(0, N, size=N//4)] -= 1.0

            try:
                res = minimize(
                    lambda x: _objective_full(x, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = res.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # === Phase 5: Adaptive coordinate descent polish on exact objective ===
    if time.time() < deadline - 5.0:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                idx = rng.integers(0, N)
                for delta in step_sizes:
                    for sign in [1, -1]:
                        h_new = best_h.copy()
                        h_new[idx] += sign * delta
                        val_new = _evaluate_exact(h_new)
                        if val_new < best_val - 1e-14:
                            best_val = val_new
                            best_h = h_new
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                step_sizes = [s * 0.5 for s in step_sizes]
                if step_sizes[0] < 1e-6:
                    break

    # === Phase 6: Final high-precision subgradient polish ===
    if time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            h = best_h.copy()
            s = np.sum(h)
            if abs(s) < 1e-10:
                break
            c = _autoconv_fft(h)
            max_idx = np.argmax(c)
            h = h - np.mean(h) + 1.0

            base_val = _evaluate_exact(h)
            for scale in [0.01, 0.005, 0.001]:
                peak_lag = max_idx
                if peak_lag < N:
                    direction = np.zeros(N)
                    for i in range(N - peak_lag):
                        direction[i] -= h[i + peak_lag]
                        direction[i + peak_lag] -= h[i]
                else:
                    lag = peak_lag - (2*N - 2)
                    direction = np.zeros(N)
                    for i in range(N + lag):
                        direction[i] -= h[i - lag]
                        direction[i - lag] -= h[i]

                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    h_new = h + scale * direction
                    if abs(np.sum(h_new)) < 1e-10:
                        continue
                    val_new = _evaluate_exact(h_new)
                    if val_new < best_val - 1e-14:
                        best_val = val_new
                        best_h = h_new.copy()
                        break

    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
