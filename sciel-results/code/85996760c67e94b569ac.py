"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a generalized polynomial basis expansion.
    
    This approach uses a linear combination of polynomial basis functions 
    (linear, quadratic, and interaction terms) to approximate the growth rate.
    This is a flexible non-parametric-style approach that does not assume 
    specific biological mechanisms like Monod kinetics or Gaussian optima.
    
    Parameters:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    params: Array of 10 fitted constants.
        p[0]: Bias
        p[1]: Coefficient for s
        p[2]: Coefficient for s^2
        p[3]: Coefficient for temp
        p[4]: Coefficient for temp^2
        p[5]: Coefficient for pH
        p[6]: Coefficient for pH^2
        p[7]: Coefficient for s * temp
        p[8]: Coefficient for s * pH
        p[9]: Coefficient for temp * pH
    """
    p = params
    
    # Basis functions
    # Note: We assume inputs are in a reasonable range. 
    # If inputs are large, we might need scaling, but the task says inputs are in [1, 5] range roughly.
    # s in [1, 40], temp in [1, 50]?, pH in [1, 14]?, b in [1, 5].
    # Let's check the ranges provided: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12].
    # The order is b, s, temp, pH.
    # b: ~1-5
    # s: ~1-40
    # temp: ~1-50 (likely)
    # pH: ~2-12 (likely)
    
    # To improve numerical stability for BFGS, it is often better to normalize inputs.
    # However, we don't know the exact means/stds. 
    # Given the ranges, s and temp might cause large values in squared terms.
    # s^2 up to 1600. temp^2 up to 2500.
    # This might be fine for BFGS if initialized at 1.
    
    term_s = s
    term_s2 = s * s
    term_t = temp
    term_t2 = temp * temp
    term_h = pH
    term_h2 = pH * pH
    
    term_st = s * temp
    term_sh = s * pH
    term_th = temp * pH
    
    # Linear combination
    y = (p[0] + 
         p[1] * term_s + 
         p[2] * term_s2 + 
         p[3] * term_t + 
         p[4] * term_t2 + 
         p[5] * term_h + 
         p[6] * term_h2 + 
         p[7] * term_st + 
         p[8] * term_sh + 
         p[9] * term_th)
         
    return y
# EVOLVE-BLOCK-END
