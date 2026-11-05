import numpy as np
import time
from scipy.fft import dct, idct
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We parameterize h using DCT coefficients.
    # h = baseline + sum(c_i * basis_i)
    # To ensure sum(h) = 1, we can fix the DC component (index 0) or adjust it.
    # Let's use the first K DCT modes (excluding DC or including it?).
    # Standard DCT-II: basis 0 is constant.
    # If we include basis 0, it controls the mean.
    # Let's fix the mean by projecting h to have sum 1 after reconstruction.
    
    # Number of DCT coefficients to optimize.
    # Low frequencies are key for minimizing peak-to-mean.
    # Let's try K=50 coefficients.
    K = 50
    
    # Precompute DCT basis vectors for the first K modes.
    # scipy.fft.dct uses DCT-II.
    # We want the basis vectors such that h = idct(coeffs, type=2, norm='ortho')
    # Or we can just work in the coefficient space and convert to time domain.
    
    # Let's define the objective function in terms of DCT coefficients 'c'.
    # c[0] is the DC component.
    # We will enforce sum(h) = 1 by scaling h if necessary, or by fixing c[0] appropriately?
    # Easier: Reconstruct h, normalize h to sum 1, then compute value.
    
    def objective(c):
        # Reconstruct h from DCT coefficients
        # Pad c to length N with zeros
        c_full = np.zeros(N)
        c_full[:K] = c
        
        # Inverse DCT to get time domain
        # norm='ortho' ensures orthonormal basis
        h = idct(c_full, type=2, norm='ortho')
        
        # Normalize h to have sum 1
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10  # Penalty
        
        h = h / s
        
        # Compute autoconvolution
        conv = np.convolve(h, h)
        
        # We want to minimize max(conv)
        # Note: The evaluator computes abs(2*N*max(conv)/sum(h)^2).
        # Since sum(h)=1, value = 2*N*max(conv).
        # Minimizing max(conv) is equivalent.
        
        return np.max(conv)

    # Initial guess: All coefficients zero except DC?
    # If c[0] is non-zero, h is constant. max(conv) for constant h=1/N is (1/N^2)*N = 1/N?
    # conv of [1/N]*N is [1/N^2]*N at center? No.
    # conv of constant vector c of length N:
    # k=0: c^2
    # k=N-1: N*c^2
    # max is N*c^2.
    # If c=1/N, max is N*(1/N^2) = 1/N.
    # Value = 2*N*(1/N) = 2. This matches the seed.
    
    # We want to reduce this.
    # Start with a slightly perturbed constant or a low-frequency wave.
    
    best_val = np.inf
    best_h = np.ones(N) / N  # Seed
    
    # Try a few random restarts in the DCT coefficient space
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize coefficients
        # Start with DC component corresponding to mean 1/N?
        # For orthonormal DCT, basis 0 is 1/sqrt(N).
        # So c[0] = mean * sqrt(N) = (1/N)*sqrt(N) = 1/sqrt(N).
        c0 = np.zeros(K)
        c0[0] = 1.0 / np.sqrt(N)
        
        # Add small random noise to lower frequencies
        if i > 0:
            # Scale noise by frequency?
            noise = rng.normal(0, 0.1, K)
            # Decay with frequency to keep it smooth
            decay = 1.0 / (1.0 + np.arange(K) * 0.1)
            c0[1:] += noise * decay
        
        # Optimize
        try:
            res = minimize(objective, c0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_val:
                c_opt = res.x
                c_full = np.zeros(N)
                c_full[:K] = c_opt
                h_candidate = idct(c_full, type=2, norm='ortho')
                
                s = np.sum(h_candidate)
                if abs(s) > 1e-10:
                    h_candidate = h_candidate / s
                    val = 2 * N * np.max(np.convolve(h_candidate, h_candidate))
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
                        
        except Exception as e:
            # If optimization fails, ignore
            pass

    return best_h
