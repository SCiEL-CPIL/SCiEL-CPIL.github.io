"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Implements a linear B-spline basis (degree 1) with 10 basis functions.
    This basis is strictly linear in params, ensuring BFGS finds the global minimum.
    The basis functions are piecewise linear, smooth (C1), and form a partition of unity,
    providing a stable and flexible approximation for continuous functions.
    """
    n_params = 10
    # Define 12 knots for 10 linear B-splines (N + p + 1 = 10 + 1 + 1 = 12)
    # We extend the domain slightly to ensure coverage at boundaries if needed, 
    # but linspace(-2, 2, 12) is standard for internal knots if we handle boundaries carefully.
    # Actually, for B-splines, the support of B_i is [t_i, t_{i+p+1}].
    # For p=1, support is [t_i, t_{i+2}].
    # If we use t = linspace(-2, 2, 12), the first basis function B_0 has support [t_0, t_2] = [-2, -1.2].
    # The last basis function B_9 has support [t_9, t_11] = [1.2, 2].
    # This covers [-2, 2] perfectly.
    
    knots = np.linspace(-2.0, 2.0, n_params + 2)
    
    # Pre-compute denominators for efficiency
    # denom[i] = t_{i+1} - t_i
    denom = np.diff(knots)
    
    # Initialize result
    result = np.zeros_like(x, dtype=np.float64)
    
    # Compute B-spline basis functions using Cox-de Boor recursion
    # B_{i,0}(x) = 1 if t_i <= x < t_{i+1}
    # B_{i,1}(x) = (x - t_i)/(t_{i+1} - t_i) * B_{i,0}(x) + (t_{i+2} - x)/(t_{i+2} - t_{i+1}) * B_{i+1,0}(x)
    
    for i in range(n_params):
        # Term 1: contribution from B_{i,0}
        # Active where t_i <= x < t_{i+1}
        mask1 = (x >= knots[i]) & (x < knots[i+1])
        term1 = np.zeros_like(x)
        if np.any(mask1):
            term1[mask1] = (x[mask1] - knots[i]) / denom[i]
            
        # Term 2: contribution from B_{i+1,0}
        # Active where t_{i+1} <= x < t_{i+2}
        mask2 = (x >= knots[i+1]) & (x < knots[i+2])
        term2 = np.zeros_like(x)
        if np.any(mask2):
            term2[mask2] = (knots[i+2] - x[mask2]) / denom[i+1]
            
        basis_i = term1 + term2
        result += params[i] * basis_i
        
    return result
# EVOLVE-BLOCK-END
