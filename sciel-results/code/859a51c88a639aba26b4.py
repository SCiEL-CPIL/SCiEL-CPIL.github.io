import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    _t0 = time.time()
    _deadline = 145.0  # 5s margin before 150s kill
    
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural Change: Gaussian RBF model with fixed centers
    # Uses 3 Gaussian basis functions in 4D input space + linear terms + intercept
    # This captures non-monotonic dependencies and complex interactions
    # 
    # p[0]: intercept
    # p[1]: weight for RBF centered at (0.5, 0.5, 0.5, 0.5) with sigma=0.3
    # p[2]: weight for RBF centered at (0.2, 0.8, 0.6, 0.4) with sigma=0.4
    # p[3]: weight for RBF centered at (0.8, 0.2, 0.4, 0.6) with sigma=0.4
    # p[4]: b linear
    # p[5]: s linear
    # p[6]: t linear
    # p[7]: ph linear
    # p[8]: b*t interaction (key for growth: biomass x temperature)
    # p[9]: s*ph interaction (substrate x pH)
    
    # Define Gaussian RBFs with fixed centers and widths
    # Center 1: middle of input space
    c1 = np.array([0.5, 0.5, 0.5, 0.5])
    sigma1 = 0.3
    d1 = (b_norm - c1[0])**2 + (s_norm - c1[1])**2 + (t_norm - c1[2])**2 + (ph_norm - c1[3])**2
    rbf1 = np.exp(-d1 / (2 * sigma1**2))
    
    # Center 2: low b, high s, moderate-high t, low-ph
    c2 = np.array([0.2, 0.8, 0.6, 0.4])
    sigma2 = 0.4
    d2 = (b_norm - c2[0])**2 + (s_norm - c2[1])**2 + (t_norm - c2[2])**2 + (ph_norm - c2[3])**2
    rbf2 = np.exp(-d2 / (2 * sigma2**2))
    
    # Center 3: high b, low s, low t, high ph
    c3 = np.array([0.8, 0.2, 0.4, 0.6])
    sigma3 = 0.4
    d3 = (b_norm - c3[0])**2 + (s_norm - c3[1])**2 + (t_norm - c3[2])**2 + (ph_norm - c3[3])**2
    rbf3 = np.exp(-d3 / (2 * sigma3**2))
    
    y = (p[0] 
         + p[1] * rbf1 
         + p[2] * rbf2 
         + p[3] * rbf3
         + p[4] * b_norm 
         + p[5] * s_norm 
         + p[6] * t_norm 
         + p[7] * ph_norm
         + p[8] * b_norm * t_norm
         + p[9] * s_norm * ph_norm)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
