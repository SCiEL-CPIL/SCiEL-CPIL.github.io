import numpy as np

N = 400

def _score(x):
    """
    Calculate the raw value: max absolute autoconvolution divided by squared integral.
    """
    # Autoconvolution
    # np.correlate(x, x, 'full') gives lags from -(N-1) to N-1
    # c[0] corresponds to lag -(N-1)
    # c[N-1] corresponds to lag 0
    c = np.correlate(x, x, 'full')
    max_abs_c = np.max(np.abs(c))
    
    # Integral (sum)
    integral = np.sum(x)
    squared_integral = integral * integral
    
    if squared_integral < 1e-12:
        return float('inf')
        
    return max_abs_c / squared_integral

def construct():
    # Candidate 1: All ones (Seed)
    x_seed = np.ones(N)
    best_x = x_seed
    best_score = _score(x_seed)
    
    # Candidate 2: Chirp sequences
    # x_i = cos(pi * i^2 / T)
    # We search for T.
    # T determines the chirp rate.
    # If T is very large, x_i ~ 1.
    # If T is small, x_i oscillates rapidly.
    
    # We scan T from 1 to N (or N^2).
    # Let's scan T in a log scale or linear scale.
    # A "quadratic phase code" typically has T ~ N.
    
    t_values = np.linspace(1, N*N, 1000)
    
    for T in t_values:
        t = np.arange(N)
        x = np.cos(np.pi * t**2 / T)
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x
            
    # Optional: Fine polish around best T if we found a good one
    # But given the time budget, the coarse scan is safe.
    
    return best_x
