"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: low-order Chebyshev polynomials for smooth global trend
    # plus a few radial basis functions (RBFs) for local flexibility.
    # This combines the well-conditioned global fit of Chebyshev with the
    # local adaptivity of RBFs, giving BFGS a smoother, more informative landscape.
    
    # Normalize input to [-1, 1] for stable Chebyshev evaluation
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind T_n(t)
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # RBF centers evenly spaced across [-3, 3]
    # Use 3 RBFs to leave room for 5 Chebyshev terms (T0..T4) = 5 params
    # Total: 5 Chebyshev + 3 RBF weights + 2 width params = 10 params
    centers = np.array([-1.5, 0.0, 1.5])
    
    # Fixed width for RBFs to keep the problem linear in the weights,
    # which helps BFGS convergence. The last 2 params can modulate
    # a shared width scale if needed, but for stability we fix width.
    # Actually, let's use 4 Chebyshev (T0..T3) + 6 RBF-related params?
    # Or 5 Chebyshev + 5 RBF weights? That's 10.
    # Let's do: 5 Chebyshev (T0-T4) + 5 RBF weights with fixed centers and width.
    # This makes the problem linear in all 10 params, which is ideal for BFGS.
    
    centers = np.array([-2.4, -1.2, 0.0, 1.2, 2.4])
    width = 1.0  # Fixed width for RBFs
    
    # Compute RBFs: exp(-((x - c)^2) / (2 * width^2))
    rbf = np.zeros_like(x)
    for c in centers:
        rbf += params[5 + int(np.argmin(np.abs(centers - c)))] * np.exp(-((x - c) ** 2) / (2.0 * width ** 2))
    
    # Wait, that indexing is messy. Let's just compute all 5 RBFs and sum with params[5:10]
    rbf_terms = np.zeros_like(x)
    for i, c in enumerate(centers):
        rbf_terms += params[5 + i] * np.exp(-((x - c) ** 2) / (2.0 * width ** 2))
    
    # Combine Chebyshev and RBF terms
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         rbf_terms)
    
    return y
# EVOLVE-BLOCK-END
