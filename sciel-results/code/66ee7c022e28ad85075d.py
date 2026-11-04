import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    """
    Construct the sequence of step heights using a parameterized waveform
    (tanh of cosine) to minimize the maximum overlap.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    i = np.arange(N)
    
    # Parameters: K (sharpness), phi (phase)
    # h_i = 0.5 + 0.5 * tanh(K * cos(2*pi*i/N + phi))
    
    def generate_h(params):
        K, phi = params
        # Ensure K is positive
        K = abs(K)
        return 0.5 + 0.5 * np.tanh(K * np.cos(2.0 * np.pi * i / N + phi))

    def loss(params):
        h = generate_h(params)
        # h is already in [0,1] and sums to N/2 (due to symmetry of tanh/cos)
        
        # Calculate C_k
        C = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(h, -k)
            C[k] = np.sum(h * (1.0 - shifted))
            
        # Soft-max loss
        beta = 20.0
        C_max = np.max(C[1:])
        if C_max == 0:
            return 0.0
        exp_C = np.exp(beta * (C[1:] - C_max))
        sum_exp = np.sum(exp_C)
        loss_val = C_max + np.log(sum_exp) / beta
        
        return loss_val

    # Initial guess
    # A square wave (K->inf) or a sine wave (K->0)
    # Let's start with a moderately sharp wave
    x0 = np.array([2.0, 0.0])
    
    # Bounds
    bounds = [(0.0, 10.0), (0.0, 2.0 * np.pi)]
    
    res = minimize(
        loss, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    h = generate_h(res.x)
    
    # Verify sum and bounds
    # Due to floating point, sum might be slightly off
    h_sum = np.sum(h)
    if abs(h_sum - N/2.0) > 1e-9:
        h += (N/2.0 - h_sum) / N
    h = np.clip(h, 0.0, 1.0)
    
    return h
