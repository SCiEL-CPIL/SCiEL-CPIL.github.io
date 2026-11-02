"""Spectral amplitude shaping strategy for minimizing autoconvolution peak."""
import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Autoconvolution (linear)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Define the objective in the spectral domain
    # We represent h by its DFT magnitudes |H[k]|.
    # To maximize the central peak of autocorrelation for a given spectrum, we assume zero phase.
    # h[n] = IDFT(|H[k]|)
    # We must ensure h[n] >= 0. We project to non-negative in the time domain.
    # The objective is to minimize the peak of the *linear* autoconvolution.
    # Note: Circular autoconvolution peak is related to max |H[k]|^2.
    # Linear autoconvolution peak is roughly related to the max of the circular one, 
    # but edge effects matter. However, minimizing the max |H[k]|^2 (spectral flatness)
    # tends to minimize the time-domain peak as well (Bennett's inequality type intuition).
    
    def spectral_objective(log_mags):
        # log_mags are the parameters
        mags = np.exp(log_mags)
        
        # Ensure mags are real and non-negative
        # DFT of real h has symmetry: H[k] = conj(H[N-k])
        # If we fix phase to 0, H[k] is real.
        # For h to be real, H must be conjugate symmetric.
        # If phase is 0, H[k] must be real. 
        # If we set H[k] = mags[k] for all k, and mags is symmetric (mags[k] = mags[N-k]), 
        # then IDFT is real.
        
        # We parameterize the magnitudes.
        # To maintain symmetry, we can just take the magnitude of the first half and mirror?
        # Or simpler: assume mags is symmetric.
        # Let's just use the full mags vector but enforce symmetry in the calculation?
        # Actually, IDFT of a real, even spectrum is real and even.
        # Let's define H[k] = mags[k].
        # If mags is symmetric (mags[k] == mags[N-k]), then h is real.
        # Let's enforce mags = 0.5 * (mags + mags[::-1]) ? No, N is even.
        # mags[0] is DC. mags[N/2] is Nyquist.
        
        # Let's just compute h = IDFT(mags).
        # If mags is not symmetric, h will be complex.
        # We take the real part.
        h_complex = np.fft.ifft(mags)
        h = np.real(h_complex)
        
        # Project to non-negative
        h = np.maximum(0, h)
        
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
            
        # Compute linear autoconvolution
        # For speed, we can use FFT based convolution if N is large, but N=600 is small.
        # np.convolve is fine.
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        
        # Normalize
        score = max_conv / (s * s)
        
        # Add a penalty if h is not "real" enough? 
        # The imaginary part of h_complex indicates loss of symmetry.
        # We want the phase to be effectively zero.
        # If we just take real part, we lose energy.
        # A better approach: enforce symmetry on mags.
        # mags_sym = 0.5 * (mags + np.roll(mags, -1, axis=0)?) No.
        # mags_sym[k] = 0.5 * (mags[k] + mags[(N-k)%N])
        
        return score

    # Better spectral parameterization:
    # Let H[k] = A[k] * exp(i*phi[k]). We fix phi[k]=0.
    # So H[k] = A[k].
    # For h to be real, A[k] must be symmetric: A[k] = A[N-k].
    # We parameterize A[k] for k=0..N/2 and mirror.
    
    def spectral_objective_sym(log_A_half):
        # log_A_half has size N//2 + 1
        A_half = np.exp(log_A_half)
        
        # Construct full A
        A = np.zeros(N)
        A[:N//2+1] = A_half
        A[N//2:] = A_half[1:-1][::-1]
        
        # H = A (phase 0)
        h_complex = np.fft.ifft(A)
        h = np.real(h_complex)
        
        # Project to non-negative
        h = np.maximum(0, h)
        
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
            
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        score = max_conv / (s * s)
        
        return score

    # Initial guess: Flat spectrum (A[k] = constant)
    # This gives a delta-like function in time (Dirac comb), but with finite N, it's a sinc-like pulse.
    # Actually, flat spectrum -> h is a delta at 0 (if centered) or spread out?
    # IDFT of constant is delta at 0.
    # If we shift the phase, we can center it. But we fixed phase to 0.
    # IDFT of constant A is delta at n=0.
    # h[0] = 1, h[n!=0] = 0.
    # Autoconvolution of delta is delta. Peak = 1. Sum = 1. Score = 1.
    # We want to minimize the peak. So we want to spread the mass.
    # Spreading mass in time domain corresponds to smoothing the spectrum?
    # No, spreading in time corresponds to *narrowing* the spectrum?
    # Wait. Uncertainty principle.
    # If h is wide (flat top), H is narrow (sinc-like).
    # If h is narrow (delta), H is wide (flat).
    # We want to minimize the peak of h*h.
    # h*h peak is minimized when h is as "flat" as possible?
    # If h is uniform (1s), h*h is a triangle. Peak = N. Sum = N. Score = N/N^2 = 1/N.
    # If h is delta, h*h is delta. Peak = 1. Sum = 1. Score = 1.
    # So uniform is better (1/600 = 0.0016) than delta (1.0).
    # Can we do better than uniform?
    # If h is a "box" of width W, h*h is a triangle of height W. Sum = W. Score = W/W^2 = 1/W.
    # So wider box is better.
    # Max width is N. Score 1/N.
    # Is it possible to get score < 1/N?
    # For non-negative h, the max of h*h is at least (sum h)^2 / (2N-1)?
    # Actually, if h is uniform, the convolution is a triangle.
    # The peak is at the center.
    # For N=600, uniform gives 1/600.
    
    # Let's try to optimize the spectrum to find a shape that beats uniform.
    # Uniform spectrum? No, uniform time domain corresponds to sinc spectrum.
    # We want to optimize A[k].
    
    # Start with A corresponding to uniform h.
    h_uniform = np.ones(N)
    H_uniform = np.fft.fft(h_uniform)
    A_uniform = np.abs(H_uniform)
    # H_uniform[0] = N. Others are 0 (approx).
    # So A_uniform is peaked at DC.
    
    # Let's start with a "flattened" spectrum to see if we can get a better time shape.
    # Or just start with random A and optimize.
    
    # We will optimize log_A_half.
    size_half = N // 2 + 1
    
    # Initial A: Let's try a "cosine" spectrum?
    # Or just start with A_uniform and perturb?
    # Or start with flat A (which gives delta h, bad) and move towards peaked A (uniform h, good)?
    # Actually, uniform h has A[0]=N, A[k!=0]=0.
    # So log_A_half should be large at 0, small elsewhere.
    
    log_A0 = np.zeros(size_half)
    log_A0[0] = np.log(N) # DC term
    # Others 0 -> exp(0)=1.
    # This is not uniform h. This is a mix of delta and uniform?
    # H[0]=N, H[k]=1.
    # h[0] = (N + N-1)/N = 1 + (N-1)/N ~ 2.
    # h[n] = 1/N.
    # This is a "box + spike".
    
    # Let's just use L-BFGS-B on log_A_half.
    
    def objective_wrapper(x):
        if time.time() > deadline:
            return 1e10
        return spectral_objective_sym(x)
    
    # Bounds for log_A: [log(1e-6), log(N*10)]
    lower_bound = np.full(size_half, np.log(1e-6))
    upper_bound = np.full(size_half, np.log(N * 10))
    bounds = list(zip(lower_bound, upper_bound))
    
    # Initial guess: Uniform h spectrum
    # A_uniform is N at 0, 0 elsewhere.
    # log_A_uniform[0] = log(N), others -inf.
    # We can't use -inf. Let's use a small value.
    log_A_init = np.full(size_half, np.log(1e-3))
    log_A_init[0] = np.log(N)
    
    best_score = compute_score(np.ones(N))
    best_h = np.ones(N).copy()
    
    # Run optimization
    try:
        res = optimize.minimize(
            objective_wrapper,
            log_A_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
            log_A_opt = res.x
            A_opt_half = np.exp(log_A_opt)
            A_opt = np.zeros(N)
            A_opt[:N//2+1] = A_opt_half
            A_opt[N//2:] = A_opt_half[1:-1][::-1]
            
            h_complex = np.fft.ifft(A_opt)
            h_opt = np.real(h_complex)
            h_opt = np.maximum(0, h_opt)
            
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
    except Exception as e:
        pass

    # Try a few random restarts in spectral space
    if time.time() < deadline - 30:
        for i in range(10):
            if time.time() > deadline:
                break
            
            # Random log_A
            log_A_rand = np.random.uniform(np.log(1e-3), np.log(N), size_half)
            # Ensure DC is significant to keep sum non-zero
            log_A_rand[0] = np.log(N) + np.random.uniform(-1, 1)
            
            try:
                res = optimize.minimize(
                    objective_wrapper,
                    log_A_rand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                log_A_opt = res.x
                A_opt_half = np.exp(log_A_opt)
                A_opt = np.zeros(N)
                A_opt[:N//2+1] = A_opt_half
                A_opt[N//2:] = A_opt_half[1:-1][::-1]
                
                h_complex = np.fft.ifft(A_opt)
                h_opt = np.real(h_complex)
                h_opt = np.maximum(0, h_opt)
                
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    return best_h.tolist()
# EVOLVE-BLOCK-END
