"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    # Universal approximator: 5 Gaussian Radial Basis Functions
    # Each RBF has a center, width, and amplitude (3 params each = 15 params)
    # But we only have 10 params total.
    # Strategy: Use 5 RBFs with shared widths to reduce parameter count?
    # Or use 3 RBFs with 3 params each + 1 global scale + 1 offset = 10?
    # Let's try: 4 RBFs with 2 params each (center, amplitude) + shared width + offset + scale
    # Actually, let's stick to the directive: 5 Gaussian RBFs.
    # If we use 5 RBFs, we need 5 centers, 5 widths, 5 amplitudes = 15 params.
    # We only have 10.
    
    # Alternative interpretation: "5 Gaussian Radial Basis Functions" might mean
    # a specific structure that fits in 10 params.
    # Perhaps: 2 RBFs with 3 params each (6) + 4 polynomial terms (4) = 10?
    # Or: 5 RBFs with fixed centers and widths, only amplitudes learnable? No, that's just 5 params.
    
    # Let's re-read: "5 Gaussian Radial Basis Functions with learnable centers, widths, and amplitudes (using 10 parameters total)"
    # This is mathematically impossible for 5 independent RBFs (3*5=15).
    # Maybe it means 3 RBFs? 3*3 = 9 params + 1 bias = 10.
    # Or maybe 2 RBFs (6 params) + 4 other terms?
    
    # Given the constraint of 10 params, let's implement a hybrid that maximizes flexibility:
    # 2 Gaussian RBFs (center, width, amplitude) = 6 params
    # 1 Linear term (slope) = 1 param
    # 1 Offset = 1 param
    # 2 Higher-order polynomial terms for global shape = 2 params
    # Total = 10.
    
    # Let's try this structure:
    # p0, p1, p2: RBF 1 (center, width, amplitude)
    # p3, p4, p5: RBF 2 (center, width, amplitude)
    # p6: linear slope
    # p7: offset
    # p8: quadratic coeff
    # p9: cubic coeff
    
    c1 = params[0]
    w1 = params[1]
    a1 = params[2]
    
    c2 = params[3]
    w2 = params[4]
    a2 = params[5]
    
    # Ensure widths are positive to avoid numerical issues with exp
    # Use softplus or absolute value with a small epsilon
    w1_safe = np.maximum(w1, 0.1)
    w2_safe = np.maximum(w2, 0.1)
    
    rbf1 = a1 * np.exp(-((x - c1) ** 2) / (2 * w1_safe ** 2))
    rbf2 = a2 * np.exp(-((x - c2) ** 2) / (2 * w2_safe ** 2))
    
    poly = params[6] * x + params[7] + params[8] * x**2 + params[9] * x**3
    
    y = rbf1 + rbf2 + poly
    
    return y
# EVOLVE-BLOCK-END
