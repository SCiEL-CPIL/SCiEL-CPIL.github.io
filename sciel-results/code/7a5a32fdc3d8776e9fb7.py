"""Symbolic regression with 1 input(s) using a hybrid basis: global trends + local RBFs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: global functional forms for overall shape + local RBFs for features.
    
    The global terms (constant, linear, quadratic, sin, cos, exp, log, 1/x) capture
    broad trends. The RBFs (with fixed centers and moderate bandwidths) provide local
    flexibility to capture non-polynomial features like bumps, sharp transitions, or
    localized oscillations.
    
    This gives BFGS a rich but well-conditioned search space: the global terms are
    smooth and well-scaled, while the RBFs are localized so they don't interfere
    with each other or the global terms too much.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
    
    # --- Global trend basis (well-scaled to be O(1) across the domain) ---
    x_c = (x - 3.05) / 3.05  # centered and scaled to ~[-1, 1]
    
    # b0: constant
    b0 = np.ones_like(x)
    
    # b1: centered linear
    b1 = x_c
    
    # b2: centered quadratic
    b2 = x_c * x_c
    
    # b3: sin(x)
    b3 = np.sin(x)
    
    # b4: cos(x)
    b4 = np.cos(x)
    
    # b5: exp(x) scaled to be O(1) at center
    b5 = np.exp(x - 3.05) / np.exp(1.5)  # scale down to avoid huge values
    
    # b6: exp(-x) scaled
    b6 = np.exp(-x) * 20.0  # exp(-6)~0.0025, *20 gives ~[0.05, 20]
    
    # b7: log(x) centered
    b7 = np.log(x) - np.log(3.05)
    
    # b8: 1/x centered and scaled
    b8 = (1.0 / x - 1.0 / 3.05) * 3.05
    
    # --- Local RBF basis (moderate bandwidths for smooth local features) ---
    # b9: Gaussian RBF centered at x=3.0 (center of domain)
    b9 = np.exp(-((x - 3.0)**2) / (2 * 0.8**2))
    
    # Stack into basis matrix
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Compute linear combination
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
