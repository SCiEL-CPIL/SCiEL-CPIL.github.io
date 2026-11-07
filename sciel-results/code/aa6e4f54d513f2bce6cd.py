# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear using first two params
        p = np.asarray(params, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        if p.shape[0] >= 2:
            return p[0] * x_arr + p[1]
        return np.zeros_like(x_arr)
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Strategy: Linear combination of 10 RBFs at fixed, evenly spaced centers.
    # This is a "Linear" model in the amplitudes, which BFGS handles very well.
    # The only non-linear parameter is the shared width (sigma).
    
    # Parameter mapping:
    # p[0]: log_sigma (log of the shared width)
    # p[1..9]: Amplitudes (a1..a9) for the first 9 RBFs
    # p[9]: Bias term (or amplitude for the 10th RBF? Let's use p[1..9] for 9 amps and p[9] for bias, 
    #        OR use p[1..9] for 9 amps and hardcode a 10th? 
    #        The task says params length is 10. 
    #        Let's use:
    #        p[0]: log_sigma
    #        p[1]: a1
    #        ...
    #        p[9]: a9
    #        We have 9 amplitudes. We can add a bias term by including a constant RBF or just adding p[9] as bias?
    #        If we use p[1..9] as 9 amps, we don't have a bias. 
    #        Let's define 10 RBFs? No, we only have 9 slots for amps if p[0] is sigma.
    #        Let's use 9 RBFs + 1 Bias.
    #        Centers: evenly spaced in [-3, 3]. 9 centers.
    #        x is in [-3, 3].
    
    log_sigma = p[0]
    # Use softplus-like or exp to ensure sigma > 0. 
    # exp(log_sigma) is simple. If log_sigma is large, sigma is large (flat). If small, sigma is small (spiky).
    # To avoid underflow/overflow, clip log_sigma.
    log_sigma = np.clip(log_sigma, -5, 5)
    sigma = np.exp(log_sigma)
    
    # Minimum sigma to prevent division by zero or extreme spikiness causing numerical issues
    min_sigma = 1e-3
    sigma = np.maximum(sigma, min_sigma)
    
    # Amplitudes
    amps = p[1:10]  # 9 amplitudes
    
    # Bias
    # We don't have a spare param if we use p[1..9] for amps. 
    # Let's adjust: Use p[0] for log_sigma. 
    # Use p[1..8] for 8 amplitudes? 
    # Or better: Use all 10 params for amplitudes of 10 RBFs, and estimate sigma via a separate mechanism? 
    # No, sigma must be a param.
    # Let's stick to 9 RBFs + Bias. 
    # Wait, if I use p[1..9] for 9 amps, where is the bias?
    # I can add a constant term 1.0 as the 10th "RBF" with amplitude p[9]? 
    # But p[9] is already used if I slice p[1:10]. 
    # Slice p[1:10] gives indices 1, 2, ..., 9. That is 9 elements.
    # So p[1], p[2], ..., p[9] are the 9 amplitudes.
    # There is no bias. 
    # A function like y = x^2 has a non-zero mean. Without a bias, the model might struggle to fit the offset.
    # However, RBFs with large sigma can approximate a constant. 
    # Alternatively, I can use p[0] for bias, p[1] for log_sigma, and p[2..9] for 8 amps?
    # 8 RBFs might be too coarse.
    
    # Let's try: 
    # p[0]: Bias
    # p[1]: log_sigma
    # p[2..9]: 8 Amplitudes
    # This gives 8 RBFs. 
    # Centers for 8 RBFs in [-3, 3]: 
    # np.linspace(-3, 3, 8)
    
    bias = p[0]
    log_sigma = p[1]
    amps = p[2:10]  # 8 amplitudes
    
    log_sigma = np.clip(log_sigma, -5, 5)
    sigma = np.exp(log_sigma)
    sigma = np.maximum(sigma, 1e-3)
    
    # Define 8 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 8)
    
    # Compute RBFs
    # RBF_i(x) = exp(-0.5 * ((x - c_i) / sigma)^2)
    # Shape: x is (N,), centers is (8,)
    # diff: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent
    exponent = -0.5 * (diff / sigma)**2
    
    # To avoid overflow in exp (though exponent is <= 0, so exp <= 1), it's safe.
    # But if sigma is very small, diff/sigma can be large, exponent very negative, exp -> 0.
    rbf_values = np.exp(exponent)
    
    # Linear combination
    # result = sum(amps[i] * rbf_values[:, i]) + bias
    result = rbf_values @ amps + bias
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs
        return p[0] * x + p[1]
        
    return result
# EVOLVE-BLOCK-END
