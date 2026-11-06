"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def coefficients(stages):
    """4th-order symmetric splitting with 6 substeps mapped to 12 stages."""
    # 4th-order symmetric splitting coefficients for 3 substeps
    theta = 2.0 ** (1.0 / 3.0)
    alpha = 1.0 / (2.0 * (1.0 - theta))
    beta = -theta / (2.0 * (1.0 - theta))
    
    # Map to 12 stages (6 substeps)
    a = [
        alpha / 6.0,
        alpha / 6.0,
        beta / 6.0,
        beta / 6.0,
        beta / 6.0,
        alpha / 6.0,
        alpha / 6.0,
        beta / 6.0,
        beta / 6.0,
        beta / 6.0,
        alpha / 6.0,
        alpha / 6.0
    ]
    b = [1.0 / 12.0] * 12
    
    return a, b
# EVOLVE-BLOCK-END
