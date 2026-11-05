"""
Combined program for the third autocorrelation inequality task.

Strategy:
- Represent h as a DFT-based sum of a few sinusoids (DC + harmonics) to allow rich, smooth, symmetric-ish shapes.
- Use differential evolution for global search over the low-dimensional parameter vector.
- Refine with L-BFGS-B in the same parameter space, using a smooth-max surrogate for differentiability.
- Final fallback: direct L-BFGS-B in h-space if time permits.
- Deadline guard: return best found well before 150 s.
"""

import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin

    def compute_value(h):
        """Exact objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.02):
        """Soft max via log-sum-exp for differentiability."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def dft_heights(params):
        """
        Build h from DFT parameters:
        params = [dc, a1, p1, a2, p2, ..., aK, pK]
        h[i] = dc + sum_k a_k * cos(2*pi*k*i/N + p_k)
        """
        K = (len(params) - 1) // 2
        i = np.arange(N, dtype=float)
        h = np.full(N, params[0], dtype=float)
        for k in range(1, K + 1):
            a = params[1 + 2 * (k - 1)]
            p = params[2 + 2 * (k - 1)]
            h += a * np.cos(2 * np.pi * k * i / N + p)
        return h

    def dft_objective(params, tau=0.02):
        """Differentiable objective using smooth max."""
        h = dft_heights(params)
        if not np.all(np.isfinite(h)):
            return 1e10
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def dft_objective_exact(params):
        """Exact objective for final evaluation."""
        h = dft_heights(params)
        if not np.all(np.isfinite(h)):
            return np.inf
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # should be 2.0

    # Phase 1: Differential Evolution in DFT parameter space
    K = 8  # number of harmonics
    n_params = 1 + 2 * K
    bounds = [(-3.0, 3.0)]  # DC
    for _ in range(K):
        bounds.append((-2.0, 2.0))   # amplitude
        bounds.append((0.0, 2.0 * np.pi))  # phase

    best_params = None
    try:
        from scipy.optimize import differential_evolution
        de_result = differential_evolution(
            dft_objective,
            bounds,
            maxiter=60,
            popsize=20,
            seed=42,
            tol=1e-7,
            polish=False,
            init='latinhypercube',
            time_limit=70
        )
        if de_result.fun < best_val:
            best_params = de_result.x.copy()
            h_candidate = dft_heights(best_params)
            val_candidate = compute_value(h_candidate)
            if val_candidate < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-6:
                best_h = h_candidate.copy()
                best_val = val_candidate
                print(f"DE: improved to {best_val:.6f}", file=sys.stderr)
    except Exception as e:
        print(f"DE error: {e}", file=sys.stderr)

    # Phase 2: L-BFGS-B refinement in DFT parameter space
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import minimize
            # Multiple restarts
            rng = np.random.RandomState(123)
            for restart in range(5):
                if time.time() > deadline - 25.0:
                    break
                if restart == 0 and best_params is not None:
                    x0 = best_params.copy()
                elif restart == 1:
                    # Perturb seed
                    x0 = np.zeros(n_params)
                    x0[0] = 1.0  # DC = 1
                    x0[1::2] = rng.uniform(-0.5, 0.5, K)
                    x0[2::2] = rng.uniform(0, 2 * np.pi, K)
                else:
                    x0 = np.zeros(n_params)
                    x0[0] = rng.uniform(0.5, 2.0)
                    x0[1::2] = rng.uniform(-1.0, 1.0, K)
                    x0[2::2] = rng.uniform(0, 2 * np.pi, K)

                result = minimize(
                    dft_objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = dft_heights(result.x)
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    best_params = result.x.copy()
                    print(f"L-BFGS-B restart {restart}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 3: Direct L-BFGS-B in h-space (final polish)
    if time.time() < deadline - 15.0:
        try:
            from scipy.optimize import minimize

            def h_objective(h):
                if not np.all(np.isfinite(h)):
                    return 1e10
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = smooth_max(conv, tau=0.01)
                return 2.0 * N * sm / (s * s)

            result = minimize(
                h_objective,
                best_h.copy(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Direct L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS-B error: {e}", file=sys.stderr)

    # Final safety checks
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    s = np.sum(best_h)
    if abs(s) < 1e-6:
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
