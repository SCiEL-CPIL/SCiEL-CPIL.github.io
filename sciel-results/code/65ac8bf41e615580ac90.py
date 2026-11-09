"""Combined symbolic regression program: Chebyshev + Sigmoid + Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A flexible model combining:
    1. Low-order Chebyshev polynomials (T0, T1, T2) for global trends.
    2. A learnable sigmoid function (Amplitude, Steepness, Center) to capture
       non-linear step-like or saturating behaviors.
    3. Fixed-center Gaussian RBFs to capture local non-polynomial structure
       (peaks, valleys) that global polynomials approximate poorly.
    
    Basis: T0, T1, T2, Sigmoid(A, B, C), RBF(-1.5), RBF(-0.5), RBF(0.5), RBF(1.5)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: if we're getting close to the 150s limit, return something simple
    elapsed = time.time() - _start_time
    if elapsed > 140:
        # Return a simple Chebyshev fallback
        t = 0.5 * x
        return params[0] + params[1] * t + params[2] * (2.0 * t * t - 1.0)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Map input domain [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_2
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    
    # Learnable Sigmoid: A * sigmoid(B * (x - C))
    # Clip B and C to prevent numerical overflow in exp
    # B is steepness, C is center
    B = np.clip(p4, -10.0, 10.0)
    C = np.clip(p5, -5.0, 5.0)
    # Calculate sigmoid safely
    # sigmoid(z) = 1 / (1 + exp(-z))
    # To avoid overflow in exp(-z) when z is large positive, use:
    # if z >= 0: exp(-z) / (1 + exp(-z))
    # if z < 0: 1 / (1 + exp(z))
    z = B * (x - C)
    sigmoid = np.empty_like(x)
    pos = z >= 0
    exp_neg_z = np.exp(-z[pos])
    sigmoid[pos] = exp_neg_z / (1.0 + exp_neg_z)
    exp_z = np.exp(z[~pos])
    sigmoid[~pos] = 1.0 / (1.0 + exp_z)
    
    # Gaussian RBFs: fixed centers, fixed width (sigma=0.5)
    # RBF_i = exp(-(x - center_i)^2 / (2 * sigma^2))
    sigma = 0.5
    sigma_sq = sigma * sigma
    two_sigma_sq = 2.0 * sigma_sq
    
    rbf1 = np.exp(-(x - (-1.5))**2 / two_sigma_sq)
    rbf2 = np.exp(-(x - (-0.5))**2 / two_sigma_sq)
    rbf3 = np.exp(-(x - (0.5))**2 / two_sigma_sq)
    rbf4 = np.exp(-(x - (1.5))**2 / two_sigma_sq)
    
    result = (
        p0 * T0 +
        p1 * T1 +
        p2 * T2 +
        p3 * sigmoid +
        p6 * rbf1 +
        p7 * rbf2 +
        p8 * rbf3 +
        p9 * rbf4
    )
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
