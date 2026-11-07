"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich 10-term basis for symbolic regression with better scaling.
    
    The 10 parameters are coefficients for a linear combination of 10 basis functions:
    1. 1 (constant)
    2. x/3 (linear, scaled)
    3. x^2/10 (quadratic, scaled)
    4. x^3/30 (cubic, scaled)
    5. x^4/100 (quartic, scaled)
    6. exp(x/2)/10 (exponential growth, scaled)
    7. exp(-x/2)/10 (exponential decay, scaled)
    8. log(x + 1)/2 (logarithmic, scaled)
    9. sin(x) (sinusoidal)
    10. cos(x) (cosinusoidal)
    
    Scaling improves conditioning for BFGS optimization.
    """
    # Compute scaled basis functions
    basis = np.column_stack([
        np.ones_like(x),                    # 1. constant
        x / 3.0,                            # 2. linear (mean ~1)
        x ** 2 / 10.0,                      # 3. quadratic (mean ~1)
        x ** 3 / 30.0,                      # 4. cubic (mean ~1)
        x ** 4 / 100.0,                     # 5. quartic (mean ~1)
        np.exp(x / 2.0) / 10.0,             # 6. exp growth (range ~[1.1, 220]/10)
        np.exp(-x / 2.0) / 10.0,            # 7. exp decay (range ~[0.05, 1.1]/10)
        np.log(x + 1.0) / 2.0,              # 8. logarithmic (range ~[0.05, 1.25]/2)
        np.sin(x),                          # 9. sinusoidal
        np.cos(x)                           # 10. cosinusoidal
    ])
    
    # Linear combination of basis functions
    return basis @ params
# EVOLVE-BLOCK-END
