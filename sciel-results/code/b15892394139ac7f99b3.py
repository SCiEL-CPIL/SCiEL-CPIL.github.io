"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    
    def objective_signed(params):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    def chirp_objective(params):
        """Objective for chirp parameters: [A, B, omega, phase].
        h[i] = A + B * cos(omega * i + phase)
        """
        A, B, omega, phase = params
        i = np.arange(N)
        h = A + B * np.cos(omega * i + phase)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Global search over chirp parameters using differential evolution
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-2.0, 2.0), (-2.0, 2.0), (0.01, 2.0 * np.pi / N), (0.0, 2.0 * np.pi)]
            de_result = differential_evolution(
                chirp_objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-8,
                polish=True
            )
            if de_result.fun < best_val:
                A, B, omega, phase = de_result.x
                i = np.arange(N)
                h_chirp = A + B * np.cos(omega * i + phase)
                val_chirp = compute_value(h_chirp)
                if val_chirp < best_val and np.all(np.isfinite(h_chirp)) and abs(np.sum(h_chirp)) > 1e-6:
                    best_h = h_chirp.copy()
                    best_val = val_chirp
                    print(f"Chirp DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DE error: {e}", file=sys.stderr)
    
    # Phase 2: Local refinement using L-BFGS-B from best
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"L-BFGS-B refine: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)
    
    # Phase 3: Try multiple low-frequency initializations with L-BFGS-B
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline - 5.0:
            break
        
        k = rng.randint(1, 8)
        a = rng.uniform(-0.5, 0.5)
        b = rng.uniform(-0.5, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        h0 = 1.0 + a * np.cos(2 * np.pi * k * np.arange(N) / N + phase) + b * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        
        s0 = np.sum(h0)
        if abs(s0) < 1.0:
            h0 = h0 + (1.0 - s0) / N
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Trial {trial}: error {e}", file=sys.stderr)
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
    
# EVOLVE-BLOCK-END
