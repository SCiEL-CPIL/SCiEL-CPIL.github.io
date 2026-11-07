import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed physics-inspired basis for a nonlinear oscillator.
    The model is linear in parameters: y = T @ params.
    
    Design choices:
    - Scale x and v so that typical values are O(1), making optimal params O(0.1-1)
      which is close to BFGS's initial guess of all ones.
    - Use polynomial terms up to cubic (Duffing/Van der Pol style) plus trig terms
      to capture potential periodic/saturation behavior.
    - No data-dependent normalization: the basis is fixed, giving BFGS a clean
      convex quadratic landscape (linear least squares).
    """
    # Scale inputs to roughly [-1, 1] range
    # x range: [-0.5728, 0.5275] -> scale by 0.55
    # v range: [-0.1754, 0.2821] -> scale by 0.23
    X = x / 0.55
    V = v / 0.23
    
    # Build fixed basis (10 terms)
    # 0: constant
    # 1: X (linear restoring)
    # 2: V (linear damping)
    # 3: X^2 (quadratic)
    # 4: X^3 (cubic Duffing)
    # 5: V^2 (quadratic damping)
    # 6: V^3 (cubic damping)
    # 7: X*V (cross)
    # 8: sin(X) (periodic)
    # 9: cos(X) - 1 (periodic, centered)
    
    T = np.column_stack([
        np.ones_like(X),
        X,
        V,
        X * X,
        X * X * X,
        V * V,
        V * V * V,
        X * V,
        np.sin(X),
        np.cos(X) - 1.0
    ])
    
    return T @ params
