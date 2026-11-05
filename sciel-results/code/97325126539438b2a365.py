"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses DCT-based spectral optimization: represent h as a sum of cosine basis
    functions, then use L-BFGS-B to minimize the peak of the linear autoconvolution.
    The DCT provides an orthonormal basis that naturally captures low-frequency
    structure, and optimizing the DCT coefficients directly shapes the power
    spectrum to minimize the autocorrelation peak.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0

    def compute_value(h):
        """Compute the evaluation metric."""
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            return np.inf
        autoconv = np.convolve(h, h)
        max_conv = np.max(autoconv)
        return abs(2 * N * max_conv / (h_sum ** 2))

    # Use zero-padded FFT for efficient linear convolution
    n_fft = 2 * N  # 800

    def autoconv_from_h(h):
        """Compute linear autoconvolution efficiently via FFT."""
        H = fft(h, n=n_fft)
        autoconv_full = ifft(H * H, n=n_fft).real
        # Linear convolution of length 2N-1 is the central portion
        # np.convolve(h, h) gives indices 0..2N-2
        # ifft gives circular convolution of length n_fft=2N
        # The linear conv is autoconv_full[N-1:N-1+N] (length N) plus wrap-around
        # Actually, for zero-padded FFT of length 2N, the linear conv is:
        # autoconv_linear[k] = autoconv_full[k] for k in [0, N-1]
        # and autoconv_linear[k] = autoconv_full[k] for k in [N, 2N-2] maps to...
        # Let me just use the direct relation:
        # np.convolve(h, h) has length 2N-1
        # ifft(fft(h,2N)**2, 2N) gives circular conv of length 2N
        # The first N-1 elements and last N-1 elements of circular conv
        # correspond to the linear conv (no wrap-around in the middle)
        # Specifically: linear_conv[k] = circ_conv[k] for k=0..N-2
        # and linear_conv[N-1+k] = circ_conv[N-1+k] for k=0..N-2? 
        # Actually simpler: just extract the right portion
        # The linear autoconvolution r[k] = sum_j h[j]*h[j+k] for k=-(N-1)..N-1
        # In np.convolve(h,h), index k+N-1 corresponds to lag k
        # So np.convolve(h,h)[N-1] is lag 0 (maximum for positive signals)
        # For our purposes, we need max of np.convolve(h,h)
        # Let's compute it directly for correctness
        return np.convolve(h, h)

    def objective_from_dct_coeffs(c):
        """
        Compute the objective value from DCT coefficients.
        c[0] is the DC component (mean * sqrt(N)), c[1:] are AC components.
        We reconstruct h from DCT, normalize so sum(h) = N, then compute value.
        """
        # Reconstruct h from DCT coefficients
        # DCT-II: c[k] = sum_j h[j] * cos(pi * k * (2*j+1) / (2*N))
        # Inverse: h[j] = (1/N) * c[0] + (2/N) * sum_{k=1}^{N-1} c[k] * cos(pi * k * (2*j+1) / (2*N))
        
        # Use scipy's idct for efficiency
        from scipy.fftpack import idct
        h_recon = idct(c, type=2, norm='ortho') * np.sqrt(N)
        
        # Actually, let's be more careful with the DCT scaling
        # scipy.fftpack.dct with norm='ortho' gives:
        # c[k] = sqrt(2/N) * sum_j h[j] * cos(pi*k*(2*j+1)/(2*N)) for k>0
        # c[0] = sqrt(2/N) * (1/sqrt(2)) * sum_j h[j] = (1/sqrt(N)) * sum_j h[j]
        # So sum(h) = c[0] * sqrt(N)
        # And h[j] = idct(c, type=2, norm='ortho')
        
        h_recon = idct(c, type=2, norm='ortho')
        
        if not np.all(np.isfinite(h_recon)):
            return 1e10
        
        h_sum = np.sum(h_recon)
        if abs(h_sum) < 1e-10:
            return 1e10
        
        # Normalize so that sum = N
        h_norm = h_recon * (N / h_sum)
        
        autoconv = np.convolve(h_norm, h_norm)
        max_conv = np.max(autoconv)
        value = abs(2 * N * max_conv / (N ** 2))  # h_sum^2 = N^2
        return value

    # Strategy: Optimize DCT coefficients using L-BFGS-B
    # Start with a few random initializations, then refine the best one
    
    # Precompute the DCT matrix for fast reconstruction
    # DCT-II matrix: D[k,j] = cos(pi * k * (2*j+1) / (2*N))
    # With orthonormal scaling
    
    # Let's use a different approach: directly optimize in the time domain
    # but with a spectral penalty to encourage flat power spectrum
    
    def objective_time_domain(h):
        """
        Objective function for L-BFGS-B optimization in time domain.
        h is a flat array of length N.
        """
        if not np.all(np.isfinite(h)):
            return 1e10
        
        h_sum = np.sum(h)
        if abs(h_sum) < 1e-10:
            return 1e10
        
        # Normalize internally for evaluation, but we want to optimize
        # the shape, not the scale. Let's fix sum(h) = N by scaling.
        h_norm = h * (N / h_sum)
        
        autoconv = np.convolve(h_norm, h_norm)
        max_conv = np.max(autoconv)
        value = abs(2 * N * max_conv / (N ** 2))
        return value

    # Try multiple random initializations and optimize each
    np.random.seed(42)
    
    for trial in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random initial h with zero-mean AC + DC
        h_ac = np.random.randn(N)
        h_ac -= np.mean(h_ac)
        h_init = h_ac + 1.0  # Add DC
        
        h_sum = np.sum(h_init)
        if abs(h_sum) < 1e-10:
            continue
        
        h_init = h_init * (N / h_sum)
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                objective_time_domain,
                h_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_value:
                h_opt = result.x
                h_sum_opt = np.sum(h_opt)
                if abs(h_sum_opt) > 1e-10:
                    h_opt = h_opt * (N / h_sum_opt)
                    value_opt = objective_time_domain(h_opt)
                    if value_opt < best_value:
                        best_value = value_opt
                        best_h = h_opt.copy()
        except Exception:
            continue

    # Also try optimizing in the DCT domain for potentially better conditioning
    if time.time() < deadline - 15.0:
        from scipy.fftpack import dct, idct
        
        # Get DCT coefficients of the best solution so far
        c_best = dct(best_h, type=2, norm='ortho')
        
        # Fix the DC component (c_best[0]) to maintain the mean
        # Optimize only the AC components
        c_dc = c_best[0].copy()
        c_ac_init = c_best[1:].copy()
        
        def objective_dct(c_ac):
            """Objective for DCT AC coefficients optimization."""
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = idct(c_full, type=2, norm='ortho')
            
            if not np.all(np.isfinite(h_recon)):
                return 1e10
            
            h_sum = np.sum(h_recon)
            if abs(h_sum) < 1e-10:
                return 1e10
            
            h_norm = h_recon * (N / h_sum)
            autoconv = np.convolve(h_norm, h_norm)
            max_conv = np.max(autoconv)
            return abs(2 * N * max_conv / (N ** 2))
        
        try:
            result_dct = minimize(
                objective_dct,
                c_ac_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result_dct.fun < best_value:
                c_ac_opt = result_dct.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_dct_opt = idct(c_full_opt, type=2, norm='ortho')
                
                h_sum_dct = np.sum(h_dct_opt)
                if abs(h_sum_dct) > 1e-10:
                    h_dct_opt = h_dct_opt * (N / h_sum_dct)
                    value_dct = objective_time_domain(h_dct_opt)
                    if value_dct < best_value:
                        best_value = value_dct
                        best_h = h_dct_opt.copy()
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
