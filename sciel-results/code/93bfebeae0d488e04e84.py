import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # Deadline guard
    import time
    start_time = time.time()
    
    # Parametric model: 5 symmetric segments
    # Heights: h1, h2, h3, h2, h1
    # Widths: w1, w2, w3, w2, w1
    # Sum of widths = 600
    # h1, h2, h3 >= 0
    
    def build_h(params):
        h1, h2, h3, w1, w2 = params
        w3 = 600 - 2 * (w1 + w2)
        if w3 < 1:
            # Enforce minimum width for center
            w1 = 1
            w2 = 1
            w3 = 600 - 4
            h1, h2, h3 = 1.0, 0.5, 0.414 # Fallback
            return np.concatenate([
                np.ones(w1) * h1,
                np.ones(w2) * h2,
                np.ones(w3) * h3,
                np.ones(w2) * h2,
                np.ones(w1) * h1
            ])
        
        h = np.concatenate([
            np.ones(w1) * h1,
            np.ones(w2) * h2,
            np.ones(w3) * h3,
            np.ones(w2) * h2,
            np.ones(w1) * h1
        ])
        return h

    def score(h):
        # Compute discrete autoconvolution
        # Use FFT for speed
        fft_h = np.fft.fft(h, n=2048)
        conv = np.fft.ifft(fft_h * fft_h).real
        # The convolution is circular, but since h is zero-padded effectively by the size?
        # No, h is length 600. We need linear convolution.
        # Using FFT with n=2048 (next power of 2 > 1200) gives linear conv.
        max_conv = np.max(conv[:600]) # Only valid part?
        # Actually, the max of the autoconvolution of a non-negative function supported on [0, N)
        # is usually at lag 0.
        # But let's take max of the whole valid range.
        # The linear convolution of two length N sequences has length 2N-1.
        # The FFT result of size 2048 contains the linear conv in the first 2N-1 samples.
        valid_conv = conv[:2*N-1]
        max_conv = np.max(valid_conv)
        
        sum_h = np.sum(h)
        max_h = np.max(h)
        
        if sum_h == 0 or max_h == 0:
            return 1e10
        
        return max_conv / (sum_h * max_h)

    def objective(params):
        h = build_h(params)
        return score(h)

    # Initial guess based on analytical dip
    # Center value ~ 0.414
    # Let's try w1=150, w2=150, w3=150? No, 2*150+2*150+150 = 750 > 600.
    # w1=100, w2=100, w3=200.
    # h1=1.0, h2=0.7, h3=0.4
    
    x0 = [1.0, 0.7, 0.4, 100, 100]
    
    bounds = [
        (0.0, 1.0), # h1
        (0.0, 1.0), # h2
        (0.0, 1.0), # h3
        (1, 200),   # w1
        (1, 200)    # w2
    ]
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                   options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_params = res.x
    best_h = build_h(best_params)
    
    # Refine with a few steps of local search on the full vector?
    # Maybe just return the parametric solution.
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
