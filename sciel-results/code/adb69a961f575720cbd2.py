"""
Symbolic regression using a sum of shifted sigmoid basis functions.
The hidden function is approximated by:
y(x) = sum_{i=0}^{4} w_i / (1 + exp(-s_i * (x - c_i)))

This basis is particularly effective for functions with step-like, saturating,
or piecewise-constant behaviors. The parameters w_i (weights), s_i (steepness),
and c_i (center) are optimized via BFGS. To ensure numerical stability and
prevent overflow in the exponential, the steepness parameters are bounded
using tanh, and the centers are bounded to a reasonable range around the
input domain [-2, 2].
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s buffer before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 parameters
             params[0:5]  -> weights w_i
             params[5:10] -> steepness raw values (mapped to s_i via tanh)
             Note: Centers c_i are fixed at equally spaced points in [-2, 2]
             to reduce parameter space and improve BFGS convergence.
             Alternatively, we could use more params, but 10 is the limit.
             Let's use 5 sigmoids.
             w_i: 5 params
             s_i: 5 params
             Total: 10 params.
             Centers c_i: fixed at -2, -1, 0, 1, 2.
    """
    if x.size == 0:
        return np.array([])
    
    # Check time limit
    if time.time() - _START_TIME > TIME_LIMIT:
        return np.zeros_like(x)

    # Extract parameters
    # Weights
    w = params[0:5]
    # Steepness raw values
    s_raw = params[5:10]
    
    # Map steepness to a bounded range to avoid overflow in exp
    # tanh maps to (-1, 1). Let's scale it to [0.1, 5.0] or similar.
    # A large steepness makes the sigmoid more step-like.
    # s_i = 0.5 * tanh(s_raw) + 0.5  -> maps to (0, 1). Too small?
    # Let's try: s_i = 1 + 4 * (0.5 * tanh(s_raw) + 0.5) = 1 + 2*tanh(s_raw) + 2 = 3 + 2*tanh(s_raw)
    # This maps s_raw to (1, 5).
    # Actually, simpler: s_i = 1 + 4 * sigmoid(s_raw) ? No, sigmoid is 0..1.
    # Let's use: s_i = 1 + 4 * (1 + tanh(s_raw)) / 2 = 1 + 2 + 2*tanh(s_raw) = 3 + 2*tanh(s_raw).
    # Range: (1, 5). This is a reasonable range for steepness in [-2, 2].
    s = 3.0 + 2.0 * np.tanh(s_raw)
    
    # Fixed centers
    c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Compute sigmoids: 1 / (1 + exp(-s * (x - c)))
    # To avoid overflow in exp, we can use a stable sigmoid calculation.
    # exp(-z) where z = -s*(x-c)
    # If z is very negative, exp(z) is 0, sigmoid is 1.
    # If z is very positive, exp(z) is inf, sigmoid is 0.
    # np.exp handles inf well (returns inf), 1/(1+inf) is 0.
    # But np.exp(-1000) is 0, 1/(1+0) is 1.
    # The only issue is if z is so negative that exp(z) underflows to 0, which is fine.
    # If z is so positive that exp(z) overflows to inf, 1/inf is 0. This is fine.
    # However, numpy might warn or return nan in some edge cases?
    # Let's use a safe sigmoid:
    # sig = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    
    # x is shape (N,), c is shape (5,). We need broadcasting.
    # z[i, j] = -s[j] * (x[i] - c[j])
    # x[:, None] - c[None, :] -> shape (N, 5)
    z = -s[None, :] * (x[:, None] - c[None, :])
    
    # Stable sigmoid
    sig = np.empty_like(z)
    mask = z >= 0
    # For z >= 0: sig = 1 / (1 + exp(-z))
    # For z < 0: sig = exp(z) / (1 + exp(z))
    
    # We need to handle the calculation carefully to avoid overflow in exp.
    # For z >= 0, exp(-z) is in (0, 1]. No overflow.
    # For z < 0, exp(z) is in (0, 1). No overflow.
    
    sig[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    sig[~mask] = np.exp(z[~mask]) / (1.0 + np.exp(z[~mask]))
    
    # y = sum(w_i * sig_i)
    # w is shape (5,), sig is shape (N, 5)
    # y[i] = sum_j w[j] * sig[i, j]
    y = sig @ w
    
    return y
# EVOLVE-BLOCK-END
