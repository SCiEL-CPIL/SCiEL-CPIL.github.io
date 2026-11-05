"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Tighter safety margin before 150s limit

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

    def dft_based_objective(params):
        """Objective for DFT-based parameterization.
        params: [dc, r1, p1, r2, p2, r3, p3, r4, p4, r5, p5, r6, p6]
        h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)
        """
        dc = params[0]
        h = np.full(N, dc)
        for k in range(1, 7):
            r = params[1 + 2*(k-1)]
            p = params[2 + 2*(k-1)]
            h += r * np.cos(2*np.pi*k*np.arange(N)/N + p)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)

    def chirp_quad_objective(params):
        """Objective for quadratic-phase chirp: h[i] = A + B*cos(omega*i + beta*i^2 + phase)"""
        A, B, omega, beta, phase = params
        i = np.arange(N, dtype=float)
        h = A + B * np.cos(omega * i + beta * i**2 + phase)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Global search over DFT-based parameters using differential evolution
    if time.time() < deadline - 45.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-2.0, 2.0)]  # DC
            for k in range(1, 7):
                bounds.append((-1.5, 1.5))  # amplitude
                bounds.append((0.0, 2.0 * np.pi))  # phase
            de_result = differential_evolution(
                dft_based_objective,
                bounds,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                dc = de_result.x[0]
                h_dft = np.full(N, dc)
                for k in range(1, 7):
                    r = de_result.x[1 + 2*(k-1)]
                    p = de_result.x[2 + 2*(k-1)]
                    h_dft += r * np.cos(2*np.pi*k*np.arange(N)/N + p)
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # Phase 1b: Quadratic-phase chirp global search
    if time.time() < deadline - 35.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [
                (-2.0, 2.0),      # A
                (-2.0, 2.0),      # B
                (0.0, 2.0*np.pi/N),  # omega
                (-0.01, 0.01),    # beta (quadratic phase)
                (0.0, 2.0*np.pi)  # phase
            ]
            de_result = differential_evolution(
                chirp_quad_objective,
                bounds,
                maxiter=35,
                popsize=12,
                seed=123,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                A, B, omega, beta, phase = de_result.x
                i = np.arange(N, dtype=float)
                h_chirp = A + B * np.cos(omega * i + beta * i**2 + phase)
                val_chirp = compute_value(h_chirp)
                if val_chirp < best_val and np.all(np.isfinite(h_chirp)) and abs(np.sum(h_chirp)) > 1e-6:
                    best_h = h_chirp.copy()
                    best_val = val_chirp
                    print(f"ChirpQuad DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ChirpQuad DE error: {e}", file=sys.stderr)

    # Phase 2: Local refinement using L-BFGS-B from best with adaptive tau
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

    # Phase 3: Try multiple diverse initializations with L-BFGS-B
    rng = np.random.RandomState(42)
    for trial in range(12):
        if time.time() > deadline - 10.0:
            break

        # Mix of low-freq harmonics and chirp components
        h0 = np.ones(N) * rng.uniform(0.5, 1.5)
        for k in rng.choice([1,2,3,4,5,6,7,8], size=3, replace=False):
            a = rng.uniform(-0.4, 0.4)
            b = rng.uniform(-0.4, 0.4)
            phase = rng.uniform(0, 2 * np.pi)
            h0 += a * np.cos(2 * np.pi * k * np.arange(N) / N + phase) + b * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        
        # Add small quadratic phase component
        beta = rng.uniform(-0.005, 0.005)
        h0 += beta * np.arange(N)**2 * rng.uniform(-0.1, 0.1)

        s0 = np.sum(h0)
        if abs(s0) < 1.0:
            h0 = h0 + (1.0 - s0) / N

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
                print(f"Trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Trial {trial}: error {e}", file=sys.stderr)

    # Phase 4: Final high-precision polish with very small tau
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
