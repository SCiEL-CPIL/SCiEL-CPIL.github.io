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
    
    def fourier_to_h(fourier_coeffs):
        """Convert Fourier coefficients to time-domain signal.
        fourier_coeffs: array of length 2*K + 1
        Index 0: DC (real)
        Index 2*k: real part of bin k
        Index 2*k+1: imaginary part of bin k
        """
        K = (len(fourier_coeffs) - 1) // 2
        h = np.zeros(N)
        h[:] = fourier_coeffs[0] / N
        
        for k in range(1, K + 1):
            re = fourier_coeffs[2*k]
            im = fourier_coeffs[2*k+1]
            # Add contribution from bin k and N-k
            # h[n] += re * cos(2*pi*k*n/N) - im * sin(2*pi*k*n/N) * 2 / N? 
            # Let's use np.fft.irfft for correctness and speed if possible, 
            # but manual construction is easier to reason about for small K.
            # Actually, let's just build the spectrum and use ifft.
            pass
            
        # Efficient implementation using ifft
        spectrum = np.zeros(N // 2 + 1, dtype=complex)
        spectrum[0] = fourier_coeffs[0]
        for k in range(1, K + 1):
            spectrum[k] = fourier_coeffs[2*k] + 1j * fourier_coeffs[2*k+1]
        
        h = np.fft.irfft(spectrum, n=N)
        return h

    def fourier_objective(params):
        """Objective for Fourier coefficients."""
        h = fourier_to_h(params)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Global search over low-frequency Fourier basis
    # Use K=15 bins -> 31 parameters
    K = 15
    num_params = 2 * K + 1
    rng = np.random.RandomState(42)
    
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import minimize
            
            # Try multiple random starts in Fourier space
            for trial in range(20):
                if time.time() > deadline - 10.0:
                    break
                
                # Initialize with small random values, DC around 1
                f0 = rng.normal(0, 0.1, num_params)
                f0[0] = 1.0 + rng.normal(0, 0.1)
                
                result = minimize(
                    fourier_objective,
                    f0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                h_opt = fourier_to_h(result.x)
                val_opt = compute_value(h_opt)
                
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Fourier Trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
                    
        except Exception as e:
            print(f"Fourier search error: {e}", file=sys.stderr)
    
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
