"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm

def coefficients(stages):
    """4th-order symmetric product formula using Suzuki-Trotter coefficients composed over substeps."""
    # 4th-order Suzuki coefficients for a 3-stage symmetric step
    # x = 1/(4 - 4^(1/3)), y = -x/(2 - 2^(1/3))
    c = 4.0 ** (1.0/3.0)
    x = 1.0 / (4.0 - c)
    y = -x / (2.0 - c)
    
    # For 12 stages, use 4 repetitions of the 3-stage 4th-order block
    n_blocks = stages // 3
    a = []
    b = []
    for _ in range(n_blocks):
        a.extend([y, 1.0 - 2.0*y, y])
        b.extend([x, 1.0 - 2.0*x, x])
    
    # Pad if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    
    return a[:stages], b[:stages]
# EVOLVE-BLOCK-END
