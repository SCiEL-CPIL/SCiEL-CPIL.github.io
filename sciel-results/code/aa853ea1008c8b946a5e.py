import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Strategy: Rational Function Approximation
    # Model: y(x) = P(x) / Q(x)
    # P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    # Q(x) = p5 + p6*x + p7*x^2 + p8*x^3 + p9*x^4
    
    # This allows modeling functions with poles, asymptotes, or rapid changes
    # that polynomials (even high degree) approximate poorly (Runge phenomenon).
    
    # Extract coefficients
    a = p[:5]
    b = p[5:]
    
    # Compute P(x) and Q(x) using Horner's method for stability
    # P(x) = (((a4*x + a3)*x + a2)*x + a1)*x + a0
    # Q(x) = (((b4*x + b3)*x + b2)*x + b1)*x + b0
    
    P = a[0]
    Q = b[0]
    
    # We need to be careful with the order for Horner.
    # Usually written high to low.
    # Let's reverse a and b to go from highest degree to lowest.
    # a[4] is x^4 coeff, a[0] is x^0 coeff.
    
    P_val = np.zeros_like(x)
    Q_val = np.zeros_like(x)
    
    # Initialize with highest degree coefficient
    P_val += a[4]
    Q_val += b[4]
    
    for i in range(3, -1, -1):
        P_val = P_val * x + a[i]
        Q_val = Q_val * x + b[i]
        
    # Safety check for denominator
    # If Q_val is close to zero, the function blows up.
    # We clip the denominator to avoid division by zero and huge values
    # that might destabilize BFGS or cause overflow.
    # However, if the true function has a pole, we want large values.
    # But for optimization stability, we might want to regularize.
    # Let's use a small epsilon to prevent exact zero division.
    
    eps = 1e-8
    # Sign-preserving clip? No, just add epsilon.
    # If Q is 0, 1/eps is large.
    # If Q is small, 1/Q is large.
    
    # To prevent BFGS from wandering into regions where Q ~ 0 and causing
    # gradient explosion, we can soft-clamp the magnitude of the prediction.
    # Or, simply return 0 if the denominator is too small? No, that's discontinuous.
    
    # Let's use a safe division:
    # y = P / (Q + sign(Q)*eps) ? No, that shifts the pole.
    # Just use:
    y = P_val / (Q_val + eps * np.sign(Q_val))
    
    # If Q_val is 0, sign is 0, so Q+0 = 0. Still division by zero.
    # Better:
    # denom = Q_val
    # y = P_val / denom
    # But replace non-finite results with 0 or a large penalty?
    # The contract says: "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
    # So if we return inf, the fit fails.
    # We must ensure finite output.
    
    # Clip the denominator to avoid 0
    # If |Q| < eps, set Q = eps * sign(Q) (or just eps if Q=0)
    abs_Q = np.abs(Q_val)
    mask = abs_Q < eps
    if np.any(mask):
        # Replace small Q with eps (preserving sign if possible, but 0 has no sign)
        # If Q is 0, sign is 0. Let's just use eps.
        Q_val[mask] = eps
        
    y = P_val / Q_val
    
    # Clip y to a reasonable range to prevent overflow in MSE calculation
    # If y is 1e100, y^2 is 1e200, which is inf.
    # Max float is ~1e308. sqrt(1e308) is 1e154.
    # So clip y to [-1e150, 1e150].
    y = np.clip(y, -1e150, 1e150)
    
    return y
