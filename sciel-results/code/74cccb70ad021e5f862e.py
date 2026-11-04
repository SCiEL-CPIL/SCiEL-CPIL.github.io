import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial guess: constant vector
    h0 = np.ones(N) / N
    
    # Objective function
    def objective(h):
        # Compute linear autocorrelation using FFT
        # Zero pad to 2N to avoid circular convolution effects
        h_pad = np.zeros(2 * N)
        h_pad[:N] = h
        c = np.real(np.fft.ifft(np.fft.fft(h_pad) ** 2))
        # The linear convolution is in c[0:2*N-1]
        # We take the max of these
        return np.max(c[:2*N-1])
        
    # Constraint: sum(h) = 1
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    # Bounds: h can be any real number, but to avoid numerical issues, 
    # we can bound them. However, the problem allows negative heights.
    # Let's not bound them initially, or use wide bounds.
    bounds = [(-10.0, 10.0) for _ in range(N)]
    
    try:
        # Run optimization
        res = minimize(objective, h0, method='SLSQP', constraints=constraints, 
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success:
            h_opt = res.x
        else:
            h_opt = h0
            
        # Check time
        if time.time() > deadline:
            return h_opt
            
        # Verify and return
        return h_opt
        
    except Exception:
        # Fallback to constant vector if optimization fails
        return h0
