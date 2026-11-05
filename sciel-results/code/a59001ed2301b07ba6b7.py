"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
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
    
    def chirp_params_to_h(params):
        """Convert chirp + harmonics parameters to h vector.
        params = [A, B, omega, phase, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6]
        h[i] = 1 + A*exp(i*(B*i^2 + omega*i + phase)) + sum_k (a_k*cos(2*pi*k*i/N + b_k) + c_k*sin(...))
        We use real-valued harmonics: a_k*cos + b_k*sin for k=1..6
        """
        A, B, omega, phase = params[0], params[1], params[2], params[3]
        i = np.arange(N)
        # Chirp component (real part)
        chirp = A * np.cos(B * i**2 + omega * i + phase)
        # Harmonic components
        h = 1.0 + chirp
        for k in range(6):
            ak = params[4 + 2*k]
            bk = params[5 + 2*k]
            h += ak * np.cos(2 * np.pi * k * i / N + bk)
        return h
    
    def chirp_objective(params, tau=0.05):
        """Objective for chirp parameters."""
        h = chirp_params_to_h(params)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    from scipy.optimize import minimize
    
    # Phase 1: Global search using chirp + harmonics parameterization (16D)
    # This reduces dimensionality and allows differential evolution to find good basins
    n_chirp_trials = 8
    for trial in range(n_chirp_trials):
        if time.time() > deadline - 60.0:
            break
        
        # Random initialization in reasonable ranges
        A = rng.uniform(-0.3, 0.3)
        B = rng.uniform(-0.01, 0.01)
        omega = rng.uniform(-0.5, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        harmonics = rng.uniform(-0.2, 0.2, size=12)
        params0 = np.concatenate([[A, B, omega, phase], harmonics])
        
        try:
            # Use differential evolution for global search in low-dim space
            result = minimize(
                chirp_objective,
                params0,
                args=(0.05,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            h_opt = chirp_params_to_h(result.x)
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Chirp trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Chirp trial {trial}: error {e}", file=sys.stderr)
    
    # Phase 2: Local refinement using full 400D L-BFGS-B on best solution found
    # Use fewer iterations but tighter tolerances for polish
    if time.time() < deadline - 30.0:
        # Try a few different smooth-max temperatures for the polish
        for tau in [0.02, 0.01, 0.005]:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    objective_signed,
                    best_h,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Polish tau={tau}: improved to {best_val:.6f}", file=sys.stderr)
            except Exception as e:
                print(f"Polish tau={tau}: error {e}", file=sys.stderr)
    
    # Phase 3: Final high-precision polish with very tight tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective_signed,
                best_h,
                args=(0.001,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Final polish: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Final polish: error {e}", file=sys.stderr)
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
    
# EVOLVE-BLOCK-END
