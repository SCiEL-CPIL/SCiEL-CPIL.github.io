"""Optimized program for the third autocorrelation inequality task.

Strategy: Use Zadoff-Chu sequence (real part) multiplied by a smooth window
function to get near-zero autocorrelation, then refine with local optimization.
"""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def objective_signed(params, tau=0.05):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def zc_window_objective(params):
        """
        Objective for Zadoff-Chu * window parameterization.
        params: [u, v, A, B, width, center, phase]
        h[i] = A + B * real(ZC[i]) * window[i]
        where ZC[i] = exp(-j*pi*u*v*i*(i+1)/N)
        window[i] = exp(-0.5*((i-center)/width)^2)  (Gaussian window)
        """
        u, v, A, B, width, center, phase = params
        i = np.arange(N, dtype=float)
        
        # Zadoff-Chu sequence (real part)
        zc_phase = -np.pi * u * v * i * (i + 1) / N + phase
        zc_real = np.cos(zc_phase)
        
        # Gaussian window
        if width < 1e-6:
            window = np.ones(N)
        else:
            window = np.exp(-0.5 * ((i - center) / width) ** 2)
        
        h = A + B * zc_real * window
        
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)

    def hann_window_objective(params):
        """
        Objective for Zadoff-Chu * Hann window.
        params: [u, v, A, B, center_frac, phase]
        h[i] = A + B * real(ZC[i]) * hann_window[i]
        hann_window centered at center_frac*N with full width N
        """
        u, v, A, B, center_frac, phase = params
        i = np.arange(N, dtype=float)
        
        zc_phase = -np.pi * u * v * i * (i + 1) / N + phase
        zc_real = np.cos(zc_phase)
        
        # Hann window centered at center_frac * N
        center = center_frac * N
        # Shifted Hann: 0.5*(1+cos(2*pi*(i-center)/N))
        hann = 0.5 * (1 + np.cos(2 * np.pi * (i - center) / N))
        
        h = A + B * zc_real * hann
        
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Global search over Zadoff-Chu * Gaussian window parameters
    if time.time() < deadline - 50.0:
        try:
            from scipy.optimize import differential_evolution
            # u and v should be coprime integers for proper ZC, but allow continuous for optimization
            # Use small integer values for u, v
            bounds = [
                (0.1, 10.0),     # u (frequency parameter)
                (0.1, 10.0),     # v (chirp rate parameter)
                (-1.0, 1.0),     # A (DC offset)
                (-1.5, 1.5),     # B (amplitude)
                (5.0, 200.0),    # width (Gaussian sigma)
                (0.0, float(N)), # center
                (0.0, 2.0*np.pi) # phase
            ]
            de_result = differential_evolution(
                zc_window_objective,
                bounds,
                maxiter=60,
                popsize=18,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                u, v, A, B, width, center, phase = de_result.x
                i = np.arange(N, dtype=float)
                zc_phase = -np.pi * u * v * i * (i + 1) / N + phase
                zc_real = np.cos(zc_phase)
                if width < 1e-6:
                    window = np.ones(N)
                else:
                    window = np.exp(-0.5 * ((i - center) / width) ** 2)
                h_zc = A + B * zc_real * window
                val_zc = compute_value(h_zc)
                if val_zc < best_val and np.all(np.isfinite(h_zc)) and abs(np.sum(h_zc)) > 1e-6:
                    best_h = h_zc.copy()
                    best_val = val_zc
                    print(f"ZC-Gaussian DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC-Gaussian DE error: {e}", file=sys.stderr)

    # Phase 2: Global search over Zadoff-Chu * Hann window parameters
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [
                (0.1, 10.0),     # u
                (0.1, 10.0),     # v
                (-1.0, 1.0),     # A
                (-1.5, 1.5),     # B
                (0.0, 1.0),      # center_frac
                (0.0, 2.0*np.pi) # phase
            ]
            de_result = differential_evolution(
                hann_window_objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=123,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                u, v, A, B, center_frac, phase = de_result.x
                i = np.arange(N, dtype=float)
                zc_phase = -np.pi * u * v * i * (i + 1) / N + phase
                zc_real = np.cos(zc_phase)
                center = center_frac * N
                hann = 0.5 * (1 + np.cos(2 * np.pi * (i - center) / N))
                h_hann = A + B * zc_real * hann
                val_hann = compute_value(h_hann)
                if val_hann < best_val and np.all(np.isfinite(h_hann)) and abs(np.sum(h_hann)) > 1e-6:
                    best_h = h_hann.copy()
                    best_val = val_hann
                    print(f"ZC-Hann DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC-Hann DE error: {e}", file=sys.stderr)

    # Phase 3: Local refinement using L-BFGS-B from best with adaptive tau
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import minimize
            for tau in [0.05, 0.01, 0.001, 0.0001]:
                if time.time() > deadline - 15.0:
                    break
                result = minimize(
                    lambda x: objective_signed(x, tau=tau),
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"L-BFGS-B (tau={tau}): improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 4: Try multiple diverse initializations with L-BFGS-B
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline - 10.0:
            break

        # Generate diverse ZC-based initializations
        u = rng.randint(1, 8)
        v = rng.randint(1, 8)
        # Ensure u and v are coprime
        from math import gcd
        while gcd(u, v) > 1:
            v = rng.randint(1, 8)
        
        i = np.arange(N, dtype=float)
        zc_phase = -np.pi * u * v * i * (i + 1) / N
        zc_real = np.cos(zc_phase)
        
        # Random window
        width = rng.uniform(10, 150)
        center = rng.uniform(0, N)
        window = np.exp(-0.5 * ((i - center) / width) ** 2)
        
        A = rng.uniform(-0.5, 0.5)
        B = rng.uniform(-1.0, 1.0)
        h0 = A + B * zc_real * window
        
        # Ensure non-zero sum
        s0 = np.sum(h0)
        if abs(s0) < 0.5:
            h0 = h0 + (0.5 - s0) / N

        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda x: objective_signed(x, tau=0.03),
                h0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Trial {trial} (ZC u={u},v={v}): improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Trial {trial}: error {e}", file=sys.stderr)

    # Phase 5: Final high-precision polish with very small tau
    if time.time() < deadline - 3.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda x: objective_signed(x, tau=0.00001),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-13}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Polish: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Polish error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 1.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
