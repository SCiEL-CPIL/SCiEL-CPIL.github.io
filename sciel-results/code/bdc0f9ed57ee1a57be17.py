import numpy as np
from scipy.optimize import minimize
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear-in-parameters model with learnable basis functions.
    Uses 3 Gaussian RBFs with learnable centers, widths, and amplitudes, plus a linear trend.
    The parameter mapping ensures physical constraints (centers in [0,7], widths in [0.05,10]).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    n = x.size
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Parameter mapping
    # p[0:3]: centers -> mapped to [0.1, 6.0] via sigmoid
    # p[3:6]: widths -> mapped to [0.05, 10.0] via sigmoid
    # p[6:8]: amplitudes -> mapped to [-10, 10] via tanh
    # p[8]: linear slope -> tanh * 5
    # p[9]: intercept -> tanh * 5
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    
    def tanh_bound(z, scale=10.0):
        return scale * np.tanh(z)
    
    c1 = 0.1 + 5.9 * sigmoid(p[0])
    c2 = 0.1 + 5.9 * sigmoid(p[1])
    c3 = 0.1 + 5.9 * sigmoid(p[2])
    
    w1 = 0.05 + 9.95 * sigmoid(p[3])
    w2 = 0.05 + 9.95 * sigmoid(p[4])
    w3 = 0.05 + 9.95 * sigmoid(p[5])
    
    a1 = tanh_bound(p[6], 10.0)
    a2 = tanh_bound(p[7], 10.0)
    
    slope = tanh_bound(p[8], 5.0)
    intercept = tanh_bound(p[9], 5.0)
    
    # Compute basis
    # Widths are in the exponent: exp(-w * (x-c)^2)
    # w is width, so larger w means narrower bump.
    # Let's stick to standard RBF: exp(-((x-c)/sigma)^2)
    # So if we use exp(-w * (x-c)^2), then w = 1/sigma^2.
    # My w range [0.05, 10] means sigma in [1/sqrt(10), 1/sqrt(0.05)] = [0.3, 4.4]
    # This is reasonable for domain [0.1, 6].
    
    d1 = x - c1
    d2 = x - c2
    d3 = x - c3
    
    rbf1 = np.exp(-w1 * d1**2)
    rbf2 = np.exp(-w2 * d2**2)
    rbf3 = np.exp(-w3 * d3**2)
    
    result = a1 * rbf1 + a2 * rbf2 + (a2 * rbf3) + slope * x + intercept
    # Wait, I only have 2 amplitude params (p[6], p[7]) but 3 RBFs.
    # I need 3 amplitudes. I have 10 params.
    # c1, c2, c3 (3)
    # w1, w2, w3 (3)
    # a1, a2, a3 (3)
    # slope (1)
    # That's 10. I removed intercept?
    # Let's re-allocate:
    # p[0:3] centers
    # p[3:6] widths
    # p[6:9] amplitudes
    # p[9] slope? or intercept?
    # Linear trend usually needs slope + intercept.
    # Maybe drop one RBF?
    # Let's use 2 RBFs + Linear (slope, intercept).
    # p[0:2] centers
    # p[2:4] widths
    # p[4:6] amplitudes
    # p[6] slope
    # p[7] intercept
    # p[8:10] ? 
    # Let's use 3 RBFs and no linear trend? Or 2 RBFs + Quadratic?
    # Given the "exp decay" inspiration, maybe the function is smooth.
    # Let's try 3 RBFs + 1 global scale? No, BFGS can find scale.
    # Let's use 3 RBFs + 1 Linear Term (slope*x). Intercept can be captured by RBFs if they overlap.
    # But better to have intercept.
    # Let's go with:
    # 3 RBFs (9 params) + 1 Intercept. No slope.
    # Or 2 RBFs (6 params) + Quadratic (3 params: x^2, x, 1). Total 9. 1 spare.
    # Quadratic is a strong basis.
    # Let's try: 3 RBFs + 1 Bias.
    # If the target has a slope, RBFs can approximate it with wide bumps.
    
    a3 = tanh_bound(p[8], 10.0)
    bias = tanh_bound(p[9], 10.0)
    
    result = a1 * rbf1 + a2 * rbf2 + a3 * rbf3 + bias
    
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
