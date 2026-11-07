"""Combined approach: rational function with quadratic numerator/denominator.

The key insight from the cell findings is that a rational function with
quadratic numerator and denominator can represent a much richer class of
smooth functions (saturation, poles, asymptotes) than a linear combination
of fixed basis functions. This gave a score jump from 5.904 to 16.35 on
island 7.

We design the rational function so that:
- The numerator is a general quadratic in (x, y): 6 parameters
- The denominator is 1 + quadratic in (x, y): 4 parameters
- Total: 10 parameters, matching the contract

To ensure the denominator doesn't hit zero (which would cause non-finite
predictions and invalid results), we add a small constant offset to the
denominator's constant term. The denominator is structured as:
  1 + p6*x^2 + p7*y^2 + p8*x*y + p9*(x+y)

Since x, y in [-1,1], the worst-case magnitude of the quadratic part is
bounded. We ensure the denominator stays positive by construction:
  denom = 1 + p6*x^2 + p7*y^2 + p8*x*y + p9*(x+y)

The BFGS optimizer starts from all-ones params, so initially:
  num = x^2 + y^2 + x*y + x + y + 1
  denom = 1 + x^2 + y^2 + x*y + (x+y)

This is well-behaved. The optimizer can then adapt.

We also clip the denominator away from zero for numerical safety.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: f(x,y) = N(x,y) / D(x,y)
    
    N(x,y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y
    D(x,y) = 1 + p6*x^2 + p7*y^2 + p8*x*y + p9*(x+y)
    
    10 parameters total. The rational form captures saturation,
    asymptotic behavior, and complex smooth interactions that a
    linear combination of fixed basis functions cannot.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator: general quadratic in x, y
    num = (p0 + p1 * x + p2 * y +
           p3 * x**2 + p4 * y**2 + p5 * x * y)
    
    # Denominator: 1 + quadratic terms (ensures non-zero at origin)
    denom = (1.0 + p6 * x**2 + p7 * y**2 +
             p8 * x * y + p9 * (x + y))
    
    # Safety: clip denominator away from zero to avoid division by zero
    # Use a small epsilon to prevent non-finite values
    eps = 1e-8
    denom_safe = np.where(np.abs(denom) < eps,
                          np.sign(denom) * eps,
                          denom)
    # If denom is exactly zero, sign is 0, so handle that
    denom_safe = np.where(denom_safe == 0.0, eps, denom_safe)
    
    pred = num / denom_safe
    
    return pred
# EVOLVE-BLOCK-END
