"""
Strategy: Hybrid Trigonometric-Polynomial Basis (Linear-in-Params).

Rationale:
1. The current strategy uses a 9th-degree polynomial. While flexible for smooth non-periodic functions, it suffers from Runge's phenomenon at the boundaries of [-3, 3] and requires high degrees to approximate periodic functions like sin(x) or cos(x), leading to poor generalization if the target is oscillatory.
2. Many symbolic regression targets involve periodic or quasi-periodic components. A basis that explicitly includes low-frequency sine and cosine terms can capture these structures with very few parameters.
3. We construct a fixed basis of 10 functions:
   - 3 polynomial terms: 1, x, x^2 (to handle non-periodic trends/offsets)
   - 7 trigonometric terms: sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x), sin(4x)
   This selection covers frequencies up to ~4 rad/unit, which is sufficient for most smooth functions on [-3, 3].
4. The model remains linear in the parameters (params[0..9] are coefficients). Therefore, the MSE objective is convex quadratic in params. BFGS will converge reliably to the global minimum.
5. This hybrid basis spans a richer function class than pure polynomials of the same parameter count, specifically excelling at periodic targets while retaining the ability to fit linear/quadratic trends.
6. Numerical stability is high: trigonometric functions are bounded, and low-degree polynomials are well-conditioned on [-3, 3].
"""

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a hybrid polynomial-trigonometric basis.
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: sin(x)
    4: cos(x)
    5: sin(2x)
    6: cos(2x)
    7: sin(3x)
    8: cos(3x)
    9: sin(4x)
    
    params[0..9] are the coefficients for these basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Precompute basis vectors
    # To ensure efficiency and avoid repeated computation if called multiple times 
    # (though BFGS calls it many times, the cost is trivial for N=200), 
    # we compute the basis matrix explicitly.
    
    # Basis 0: Constant
    b0 = np.ones_like(x)
    
    # Basis 1: Linear
    b1 = x
    
    # Basis 2: Quadratic
    b2 = x * x
    
    # Basis 3-9: Trigonometric
    # Using standard numpy trig functions
    b3 = np.sin(x)
    b4 = np.cos(x)
    b5 = np.sin(2.0 * x)
    b6 = np.cos(2.0 * x)
    b7 = np.sin(3.0 * x)
    b8 = np.cos(3.0 * x)
    b9 = np.sin(4.0 * x)
    
    # Combine using dot product or explicit sum
    # Explicit sum is faster for small fixed number of terms than building a matrix and matmul
    result = (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
    
    return result
# EVOLVE-BLOCK-END
