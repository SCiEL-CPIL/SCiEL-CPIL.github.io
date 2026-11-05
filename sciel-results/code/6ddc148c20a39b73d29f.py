import numpy as np
import time
import sys

N = 400

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

    def h_from_coeffs(coeffs):
        """
        Reconstruct h from low-frequency Fourier coefficients.
        coeffs: array of shape (K, 2) containing [real, imag] for frequencies 1..K
        We force the DC component to be 1 (or some constant) to handle scaling, 
        but actually the objective is scale-invariant (numerator scales with h^2, denominator with h^2).
        However, we need sum(h) != 0. 
        Let's define h[k] = sum_{m=1}^K (c_m cos(2*pi*m*k/N) + s_m sin(2*pi*m*k/N))
        This ensures mean(h) is 0 if we don't include DC. 
        Wait, if mean(h) is 0, sum(h) is 0, which is invalid.
        So we must include a DC offset or ensure sum(h) != 0.
        Let's add a fixed DC component A0.
        h[k] = A0 + sum_{m=1}^K (c_m cos(2*pi*m*k/N) + s_m sin(2*pi*m*k/N))
        
        To keep the parameter space bounded and simple, let's use:
        h = A0 * ones(N) + real(FFT_inv(coeffs_padded))
        where coeffs_padded has 0 at DC and infinity? No.
        
        Better approach:
        Let x be the vector of real coefficients for cos and sin terms for frequencies 1..K.
        h = A0 * np.ones(N) + sum_{m=1}^K x[2m-1]*cos(2*pi*m*k/N) + x[2m]*sin(2*pi*m*k/N)
        
        We can optimize A0 and x.
        """
        K = len(coeffs) // 2 + 1 # coeffs includes A0? Let's say coeffs[0] is A0, rest are pairs.
        A0 = coeffs[0]
        k = np.arange(N)
        h = np.full(N, A0)
        
        # Number of harmonic pairs
        num_pairs = (len(coeffs) - 1) // 2
        
        for i in range(num_pairs):
            freq = i + 1
            c = coeffs[1 + 2*i]
            s = coeffs[2 + 2*i]
            h += c * np.cos(2 * np.pi * freq * k / N)
            h += s * np.sin(2 * np.pi * freq * k / N)
            
        return h

    def objective(params):
        """
        Objective function for the Fourier coefficients.
        params: [A0, c1, s1, c2, s2, ...]
        """
        h = h_from_coeffs(params)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        # The value is 2*N*max(conv)/s^2
        # To make it differentiable/smooth for SA, we can use a soft max
        # But for SA, we can just use the exact value or a slightly smoothed one.
        # Exact value is non-smooth due to max.
        # Let's use a log-sum-exp with a small temperature for the objective to guide SA.
        tau = 0.01
        conv_max = np.max(conv)
        if conv_max > 0:
             sm = conv_max + tau * np.log(np.sum(np.exp((conv - conv_max) / tau)))
        else:
             sm = conv_max # If all negative, max is negative, but we take abs later?
             # Actually max(conv) can be negative if all conv entries are negative.
             # But conv[0] = sum(h^2) >= 0. So max is always >= 0.
             
        return abs(2.0 * N * sm / (s * s))

    # Initial parameters
    # Start with a random low-frequency signal
    K_max = 20 # Use first 20 harmonics
    num_params = 1 + 2 * K_max
    best_params = np.zeros(num_params)
    best_params[0] = 1.0 # A0
    # Random small perturbations for harmonics
    np.random.seed(42)
    for i in range(K_max):
        best_params[1 + 2*i] = np.random.uniform(-0.1, 0.1)
        best_params[2 + 2*i] = np.random.uniform(-0.1, 0.1)
    
    best_h = h_from_coeffs(best_params)
    best_val = compute_value(best_h)
    
    # Seed value
    seed_h = np.ones(N)
    seed_val = compute_value(seed_h)
    if best_val > seed_val:
        best_h = seed_h
        best_val = seed_val
        # Reset params to correspond to seed? Seed is DC=1, harmonics=0
        best_params = np.zeros(num_params)
        best_params[0] = 1.0

    # Simulated Annealing
    current_params = best_params.copy()
    current_val = best_val
    T = 1.0
    T_min = 1e-6
    alpha = 0.995
    step_scale = 0.1
    
    iterations = 0
    max_iter = 50000
    
    while time.time() < deadline - 10.0 and T > T_min:
        iterations += 1
        
        # Generate neighbor
        new_params = current_params + np.random.normal(0, step_scale, size=num_params)
        
        # Evaluate
        new_val = objective(new_params)
        
        # Acceptance criterion
        if new_val < current_val or np.random.rand() < np.exp(-(new_val - current_val) / T):
            current_params = new_params
            current_val = new_val
            
            # Update best
            if current_val < best_val:
                best_val = current_val
                best_params = current_params.copy()
                # Optional: print progress
                if iterations % 5000 == 0:
                     print(f"SA Iter {iterations}: T={T:.6f}, Best={best_val:.6f}", file=sys.stderr)
        
        # Cool down
        T *= alpha
        
        # Adaptive step size: if we are stuck, increase step? 
        # Or just let T handle it.
        # Let's slowly decay step_scale too.
        step_scale *= 0.999

    # Final local refinement using L-BFGS-B on the best Fourier coefficients
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            
            def obj_lbfgs(params):
                return objective(params)
            
            result = minimize(
                obj_lbfgs,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
                print(f"Post-SA L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Post-SA L-BFGS-B error: {e}", file=sys.stderr)

    # Construct final h
    final_h = h_from_coeffs(best_params)
    final_val = compute_value(final_h)
    
    # Compare with seed
    if final_val < seed_val and np.all(np.isfinite(final_h)) and abs(np.sum(final_h)) > 1e-6:
        return final_h
    else:
        return seed_h

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(f"Result value: {compute_value(h)}")
