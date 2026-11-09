"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    deadline = time.time() + 600.0  # 630s kill, keep 30s margin
    
    # Bessel zeros of J_0: j_{0,k} ≈ (k + 1/4)π for large k, but exact first few:
    # j_{0,1}≈2.4048, j_{0,2}≈5.5201, j_{0,3}≈8.6537, ...
    # For our purpose, z_i = (j_{0,i})^2 captures the Laguerre zero distribution for α=-1/2
    # Use asymptotic: j_{0,k} ≈ (k - 1/4)π for k>=1
    # So z_i ≈ ((i - 1/4)π)^2
    
    # But we can tune: z_i = ((i + offset) * scale)^2
    # The measured best used Bessel zeros directly. Let's use the exact Bessel zero approximation.
    
    # First, try the direct Bessel zero squared values
    # j_{0,k} for k=1..12 (zeros of J_0)
    # Using the approximation j_{0,k} ≈ (k - 0.25)*π + O(1/k)
    # More accurate: j_{0,k} = (k-1/4)*π - 1/(8*(k-1/4)*π) + ...
    
    def bessel_zero_J0(k):
        """Approximate k-th zero of J_0 using asymptotic expansion."""
        # j_{0,k} ≈ (k - 1/4)*π - 1/(8*(k-1/4)*π) + 9/(128*(k-1/4)^3*π^3)
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728  # j_{0,1} exact
        return x - 1.0 / (8.0 * x) + 9.0 / (128.0 * x**3)
    
    # Initial guess: z_i = (j_{0,i})^2
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # We can refine by tuning a global scale and offset
    # z_i = ((i + a) * b)^2, fit a and b to minimize B
    # But computing B requires the full sympy solve which is expensive
    # For now, return the Bessel-zero-squared values which measured well
    
    # Ensure positive, finite, distinct
    result = []
    for z in z_bessel:
        if not math.isfinite(z) or z <= 0:
            result = None
            break
        result.append(z)
    
    if result is None or len(result) != 12:
        # Fallback
        result = [4.0 * (i + 1) for i in range(12)]
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    return result
# EVOLVE-BLOCK-END
