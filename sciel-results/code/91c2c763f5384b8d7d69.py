import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    def smooth_objective(h):
        """
        Differentiable surrogate for the objective function.
        Uses log-sum-exp to approximate max(autoconv) smoothly.
        """
        if not np.all(np.isfinite(h)):
            return float('inf')
        
        s = np.sum(h)
        if abs(s) < 1e-9:
            return float('inf')
            
        # Compute autoconvolution via FFT (much faster than np.convolve for repeated evals in gradient methods)
        # z[k] = sum_j h[j] h[k-j]
        # FFT(h) * FFT(h) -> ifft -> z
        # Note: np.fft.ifft returns complex, but for real h, imaginary part is ~0.
        # We need the real part.
        
        H = np.fft.fft(h)
        Z = np.fft.ifft(H * H)
        z = Z.real
        
        # The linear convolution length is 2N-1. FFT gives length N (circular).
        # We need to handle the wrap-around or use a larger FFT size.
        # For N=400, 2N-1 = 799. Next power of 2 is 1024.
        # Let's recompute with proper size to avoid aliasing in the max.
        # Actually, for the surrogate, we can just use the circular convolution if we assume the peak is in the center?
        # No, the peak can be anywhere.
        # Let's use a larger FFT size to get linear convolution.
        
        fft_len = 2 * N
        H_lin = np.fft.fft(h, n=fft_len)
        Z_lin = np.fft.ifft(H_lin * H_lin)
        z_lin = Z_lin.real[:2*N-1]
        
        # Smooth max: log-sum-exp(z / tau) * tau
        # To avoid overflow, subtract max
        z_max = np.max(z_lin)
        # Temperature tau. Smaller tau -> closer to max. Larger tau -> smoother.
        # Start with moderate tau, can anneal if needed, but let's fix it for simplicity.
        tau = 1.0 
        
        # If z_max is very large, exp(z/tau) might overflow.
        # Scale z by 1/tau first.
        z_scaled = z_lin / tau
        z_max_scaled = np.max(z_scaled)
        
        # log_sum_exp = z_max_scaled + log(sum(exp(z_scaled - z_max_scaled)))
        exp_terms = np.exp(z_scaled - z_max_scaled)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            log_sum_exp = z_max_scaled
        else:
            log_sum_exp = z_max_scaled + np.log(sum_exp)
            
        max_z_smooth = log_sum_exp * tau
        
        val = abs(2 * N * max_z_smooth / (s * s))
        return val

    # 1. Initial Guess
    # Start with a "bump" function that has a wide main lobe and negative side lobes.
    # This structure is known to reduce the peak-to-sidelobe ratio in autocorrelation.
    # A raised cosine or similar window.
    
    # Let's try a few different starting points and pick the best one for optimization.
    # Starting point 1: All ones (seed)
    h_init_1 = np.ones(N)
    
    # Starting point 2: Raised Cosine window (Hann window)
    # This has negative lobes? No, Hann is positive.
    # Let's try a window that goes negative.
    # e.g., h[n] = 1 - 2*(n/N - 0.5)^2? No, that's just a parabola.
    
    # Let's use a spectral approach: define H(f) and ifft.
    # We want H(f) to be concentrated but with some phase/shape to cancel peaks.
    
    # Actually, let's just use random starts and the seed, and run L-BFGS-B from the best.
    
    candidates = []
    
    # Candidate 1: Seed
    candidates.append(np.ones(N))
    
    # Candidate 2: Random Gaussian
    rng = np.random.default_rng(42)
    h_rand = rng.normal(0, 1, N)
    candidates.append(h_rand)
    
    # Candidate 3: "Box" with negative sides
    # High in middle, low on edges
    h_box = np.zeros(N)
    h_box[N//4:3*N//4] = 1.0
    h_box[:N//4] = -0.5
    h_box[3*N//4:] = -0.5
    candidates.append(h_box)
    
    # Candidate 4: Chebyshev-like oscillation
    x = np.linspace(-1, 1, N)
    h_cheb = np.cos(5 * np.arccos(x)) # T_5(x)
    candidates.append(h_cheb)
    
    best_h = None
    best_val = float('inf')
    
    # Evaluate initial candidates
    for h_cand in candidates:
        v = compute_value(h_cand)
        if v < best_val:
            best_val = v
            best_h = h_cand.copy()
            
    # 2. Optimization
    # We will run L-BFGS-B from the best initial candidate.
    # Since the objective is scale invariant, the optimizer might drift in scale.
    # L-BFGS-B handles this okay usually, but we can also normalize h in the objective?
    # No, the objective is defined on h. The gradient will push towards optimal scale.
    
    # Let's try a few iterations from different starts if time permits.
    
    starts_to_optimize = [best_h]
    
    # Add a few more diverse starts if time is available
    if time.time() < deadline - 60:
        # Add random starts
        for _ in range(5):
            h_rand = rng.normal(0, 1, N)
            starts_to_optimize.append(h_rand)
            
    # Optimize
    for h_start in starts_to_optimize:
        if time.time() > deadline - 10:
            break
            
        try:
            # L-BFGS-B
            res = minimize(
                smooth_objective, 
                h_start, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                # Verify with exact objective
                h_check = res.x
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check.copy()
        except Exception:
            pass

    # 3. Final Polish
    # One more L-BFGS-B run from the best found with tighter tolerance
    if best_h is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                smooth_objective, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.fun < best_val:
                h_check = res_final.x
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check.copy()
        except Exception:
            pass

    # Ensure validity
    if best_h is None or not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
