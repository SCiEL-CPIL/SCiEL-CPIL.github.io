import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftshift, ifftshift

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        # np.convolve(h, h) is the full linear convolution.
        # For zero-padded circular convolution, we need length 2N-1.
        # The max of linear convolution is the same as the max of circular convolution 
        # if we zero-pad to length >= 2N-1.
        # However, the problem defines it as np.convolve(h, h) which is length 2N-1.
        # Let's just use np.convolve for exactness in the final check.
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    # --- Phase 1: Spectral Design ---
    # We want to minimize the peak of the autoconvolution.
    # Autoconvolution z = h * h. 
    # In frequency domain: Z = H * H (element-wise product of DFTs).
    # Actually, circular convolution corresponds to element-wise product of DFTs.
    # Linear convolution is the first 2N-1 samples of circular convolution of zero-padded h.
    # If we zero-pad h to length L >= 2N-1, then circular conv of padded h is linear conv of h.
    # Let L = 400 (N). If we use DFT of length N, we get circular convolution.
    # The problem uses np.convolve(h, h) which is linear.
    # Note: max(linear_conv) == max(circular_conv) IF the wrap-around doesn't create larger peaks.
    # Usually, for smooth h, the peak is in the center (lag 0), which is the same in both.
    # Lag 0 value: sum(h^2).
    # Circular conv at lag k: sum h[i] h[i+k mod N].
    # Linear conv at lag k: sum h[i] h[i+k] (where indices are valid).
    # For k=0, they are identical.
    # For other k, circular includes wrap-around terms.
    # If h is smooth and decays or is periodic-like, the difference is small.
    # We will optimize based on the circular convolution peak (via DFT) as a proxy, 
    # then refine with exact linear convolution.
    
    # Representation: h = ifft(A * exp(i*Phi))
    # We fix the phase Phi to 0 for simplicity? No, h must be real.
    # If h is real, H must be conjugate symmetric.
    # Let's parameterize h directly in time domain but use a smooth surrogate 
    # that leverages frequency insights? 
    # Better: Optimize a "flat spectrum" target.
    
    # Let's try a different structural change: 
    # Use a "Bump" function in time domain but optimize its shape parameters 
    # using a robust global optimizer (Nelder-Mead) on a small set of parameters 
    # that define a superposition of cosines (Fourier series) with optimized amplitudes.
    # This is similar to the current program but using a more robust local optimizer 
    # and a specific initialization strategy: "White Noise Spectral Flattening".
    
    # Actually, the inspiration says: "Fourier-domain flat-spectrum initialization... improved island 5".
    # Let's implement that directly.
    
    # 1. Generate a random phase spectrum, set amplitude to 1 (white spectrum).
    # This creates a pseudo-random signal with flat spectrum.
    # Then, we optimize the amplitudes to minimize the time-domain peak of autoconvolution.
    
    # DFT of length N
    # H[k] = A[k] * exp(i * P[k])
    # For real h, H[N-k] = conj(H[k]).
    # We optimize A[k] for k=0..N/2.
    
    rng = np.random.default_rng(42)
    
    # Number of frequency bins to optimize (excluding DC and Nyquist if needed, but let's include all positive freqs)
    # We will optimize the log-amplitudes to keep them positive.
    K_opt = N // 2 + 1  # 0 to N/2
    
    # Initialize amplitudes to 1 (flat spectrum)
    log_A_init = np.zeros(K_opt)
    
    # Fixed random phases for non-DC, non-Nyquist
    P = rng.uniform(-np.pi, np.pi, size=N)
    # Enforce conjugate symmetry for phases? 
    # If we construct H from A and P, we must ensure H is conjugate symmetric.
    # H[k] = A[k] * exp(i P[k])
    # H[N-k] = A[N-k] * exp(i P[N-k])
    # We want H[N-k] = conj(H[k]) = A[k] * exp(-i P[k])
    # So A[N-k] = A[k] and P[N-k] = -P[k].
    
    # Let's define the optimization variables as A[k] for k=0..N/2.
    # We fix P[k] for k=1..N/2-1 to random values, and P[0]=0, P[N/2]=0 (if N even).
    
    P_opt = np.zeros(N)
    P_opt[1:N//2] = rng.uniform(-np.pi, np.pi, size=N//2 - 1)
    # P[N//2] is 0 for real signal (Nyquist is real)
    # P[0] is 0 (DC is real)
    
    def h_from_log_A(log_A):
        A = np.exp(log_A)
        # Construct full H
        H = np.zeros(N, dtype=complex)
        H[0] = A[0]
        if N % 2 == 0:
            H[N//2] = A[N//2]
        for k in range(1, N//2):
            H[k] = A[k] * np.exp(1j * P_opt[k])
            H[N-k] = A[k] * np.exp(-1j * P_opt[k])
        
        h = np.real(ifft(H))
        return h

    def objective_log_A(log_A):
        h = h_from_log_A(log_A)
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        # Use circular convolution as a fast proxy? 
        # np.convolve is O(N^2). N=400 is 160k ops. Very fast.
        # Let's use exact np.convolve to be safe.
        z = np.convolve(h, h)
        val = abs(2 * N * np.max(z) / (s * s))
        return val

    # Optimize log_A
    # Initial guess: Flat spectrum (log_A = 0)
    x0 = np.zeros(K_opt)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    try:
        res = minimize(
            objective_log_A, 
            x0, 
            method='L-BFGS-B',
            bounds=[(0, 10)] * K_opt, # Log amplitudes between e^0=1 and e^10
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res.fun < best_val:
            h_candidate = h_from_log_A(res.x)
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-9:
                val_candidate = compute_value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
    except Exception:
        pass

    # --- Phase 2: Time-Domain Refinement ---
    # The spectral optimization might get stuck in a local minimum of the smooth proxy or 
    # the discrete grid might allow for sharper features that reduce the max further.
    # We refine h directly.
    
    def smooth_objective_time(h):
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        z = np.convolve(h, h)
        # Smooth max: log-sum-exp
        z_max = np.max(z)
        # Scale z to prevent overflow in exp
        # z values are roughly sum(h^2). If h is O(1), z is O(N).
        # Let's use a temperature tau.
        tau = 1.0
        z_shifted = (z - z_max) / tau
        log_sum_exp = z_max + tau * np.log(np.sum(np.exp(z_shifted)))
        val = abs(2 * N * log_sum_exp / (s * s))
        return val

    def smooth_grad_time(h):
        # Gradient of smooth max objective
        s = np.sum(h)
        z = np.convolve(h, h)
        z_max = np.max(z)
        tau = 1.0
        z_shifted = (z - z_max) / tau
        exp_z = np.exp(z_shifted)
        sum_exp = np.sum(exp_z)
        weights = exp_z / sum_exp
        
        # Gradient of log_sum_exp wrt z is weights
        # Gradient of z wrt h is the convolution kernel.
        # z = h * h.
        # dz/dh = [h, h, ..., h] shifted.
        # Specifically, z[k] = sum_i h[i] h[i+k].
        # dz[k]/dh[j] = h[j+k] + h[j-k] (with appropriate bounds/wrapping for linear conv).
        # For linear convolution, the gradient is the autocorrelation function itself, but shifted?
        # Actually, grad_h (h*h) is the circular autocorrelation if we treat it as circular.
        # For linear, it's slightly different at the edges.
        # However, L-BFGS-B uses finite differences if gradient is not provided.
        # Given N=400, finite differences of a 400-dim vector are 400 evaluations.
        # Each eval is O(N^2) = 160k. 400 * 160k = 64M ops. 
        # In Python/NumPy, this might be slow (seconds per iteration).
        # L-BFGS-B might take 50-100 iterations. 50 * 64M = 3.2B ops. Too slow?
        # NumPy is vectorized. 160k ops is ~1ms. 400 evals = 400ms. 50 iters = 20s.
        # This is feasible within 110s budget if we limit iterations.
        
        # Let's provide the analytical gradient to be faster.
        # The gradient of the linear convolution z = conv(h, h) with respect to h
        # is not simply the autocorrelation due to boundary effects.
        # However, if we pad h with zeros to length 2N-1, the gradient of the circular convolution
        # of the padded h is the circular autocorrelation of the padded h.
        # The max is likely in the interior (lag 0 or near 0).
        # Approximation: grad_val = 2 * np.concatenate([h, np.zeros(N-1)])? No.
        
        # Let's stick to finite differences for the polish phase to avoid gradient bugs,
        # but limit the number of iterations and use a smaller step size.
        # Or, we can use a coarser grid? No, N is fixed.
        
        # Alternative: Use the fact that the max is likely at lag 0 (center of conv).
        # If max is at lag 0, z[0] = sum(h^2).
        # grad(sum(h^2)) = 2*h.
        # If the max is elsewhere, it's more complex.
        # Given the smooth nature of the spectral solution, the peak is likely flat or at lag 0.
        
        # Let's just use finite differences but with a tighter maxiter.
        return None # Let scipy use finite differences

    # Start from best_h found in Phase 1
    h_start = best_h.copy()
    
    # Ensure sum is not zero. If it is, perturb.
    if abs(np.sum(h_start)) < 1e-6:
        h_start[0] += 1.0

    try:
        res_time = minimize(
            smooth_objective_time, 
            h_start, 
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        
        if res_time.fun < best_val:
            h_candidate = res_time.x
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-9:
                val_candidate = compute_value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
    except Exception:
        pass

    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
