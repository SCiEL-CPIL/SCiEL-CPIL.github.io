"""Symbolic regression with 1 input: Adaptive RBF + Polynomial basis."""
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 140.0  # 140s safety margin before 150s limit


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of adaptive Radial Basis Functions (RBFs) 
    and a low-degree polynomial.
    
    Parameters:
    - params[0:3]: Centers of 3 RBFs (c1, c2, c3)
    - params[3:6]: Log-widths of 3 RBFs (w1, w2, w3). Actual width = exp(w)
    - params[6:10]: Coefficients for polynomial basis [1, x, x^2, x^3]
    
    This structure allows the model to adapt local features (RBFs) and 
    global trends (polynomial) simultaneously. The dependence of the basis 
    on params makes the optimization non-linear, enabling a richer search 
    space than fixed linear bases.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    n = len(x)
    
    # Extract parameters
    # 3 Centers
    c1, c2, c3 = params[0], params[1], params[2]
    
    # 3 Widths (exponentiated to ensure positivity, clamped for stability)
    # exp(params) can explode, so we clamp the exponent or the result.
    # Let's use a soft clamp: width = exp(clamp(p, -5, 5)) -> [0.006, 148]
    # Or simpler: width = 0.1 + abs(params) * 0.5? 
    # Let's stick to exp but clamp the input to exp to avoid overflow.
    w_raw = params[3:6]
    w_clamped = np.clip(w_raw, -5.0, 5.0)
    w1, w2, w3 = np.exp(w_clamped[0]), np.exp(w_clamped[1]), np.exp(w_clamped[2])
    
    # 4 Polynomial coefficients
    p0, p1, p2, p3 = params[6], params[7], params[8], params[9]
    
    # Initialize prediction with polynomial part
    pred = np.empty(n, dtype=np.float64)
    pred[:] = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Add RBF contributions
    # RBF formula: exp(-((x - c)^2) / (2 * w^2))
    # Using w^2 in denominator.
    
    # Compute distances squared
    d1_sq = (x - c1) ** 2
    d2_sq = (x - c2) ** 2
    d3_sq = (x - c3) ** 2
    
    # We need coefficients for the RBFs as well. 
    # Wait, I used all 10 params for centers, widths, and poly coeffs.
    # I forgot the RBF amplitudes!
    # A linear combination of RBFs needs amplitudes.
    # If I don't have separate amplitude params, I can't scale the RBFs independently.
    # 
    # Re-evaluating parameter allocation:
    # 10 params total.
    # Option A: 2 RBFs.
    #   2 Centers (2), 2 Widths (2), 2 Amplitudes (2), 4 Poly Coeffs (4). Total 10.
    # Option B: 3 RBFs, no Poly (just RBFs + Bias?).
    #   3 Centers (3), 3 Widths (3), 3 Amplitudes (3), 1 Bias (1). Total 10.
    # Option C: Hybrid with fixed amplitudes? No, amplitudes are crucial.
    
    # Let's go with Option A: 2 Adaptive RBFs + Cubic Polynomial.
    # This leaves room for the polynomial to handle the global shape while 
    # 2 RBFs handle local deviations.
    
    # Reset extraction
    c1, c2 = params[0], params[1]
    w1, w2 = np.exp(np.clip(params[2], -5.0, 5.0)), np.exp(np.clip(params[3], -5.0, 5.0))
    a1, a2 = params[4], params[5]
    p0, p1, p2, p3 = params[6], params[7], params[8], params[9]
    
    # Recompute
    pred = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    d1_sq = (x - c1) ** 2
    d2_sq = (x - c2) ** 2
    
    # Safety check for widths
    w1_sq = w1 * w1
    w2_sq = w2 * w2
    
    # Avoid division by zero (though exp is never 0, w_sq could be tiny if w is tiny, 
    # but we clamped log-w to -5, so w >= exp(-5) ~ 0.006, w_sq ~ 3e-5. Safe.)
    
    rbf1 = np.exp(-d1_sq / (2.0 * w1_sq))
    rbf2 = np.exp(-d2_sq / (2.0 * w2_sq))
    
    pred += a1 * rbf1 + a2 * rbf2
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
