"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible additive basis expansion.
    
    This strategy uses a purely linear-in-parameters basis expansion to ensure
    the optimization problem is convex (least squares). This allows BFGS to
    reliably find the global optimum without getting stuck in local minima,
    unlike the previous mixed linear/non-linear structure.
    
    The basis functions are chosen to capture the key biological mechanisms:
    1. Substrate saturation (Monod-like)
    2. Temperature optimum (Gaussian)
    3. pH optimum (Gaussian)
    4. Density inhibition (Exponential)
    5. Key interactions between substrate and environmental factors
    
    All parameters are linear coefficients, making the model robust for optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Bias
    # p0
    
    # 2. Substrate: Monod-like saturation s/(1+s)
    # p1 is the coefficient
    f_s = s / (1.0 + s)
    
    # 3. Temperature: Gaussian centered at 37
    # p2 is the amplitude
    # Fixed width for stability and convexity (width is not a parameter)
    f_temp = p2 * np.exp(-((temp - 37.0)**2) / 10.0)
    
    # 4. pH: Gaussian centered at 7.0
    # p3 is the amplitude
    # Fixed width for stability and convexity
    f_ph = p3 * np.exp(-((pH - 7.0)**2) / 2.0)
    
    # 5. Density: Exponential inhibition exp(-0.5*b)
    # p4 is the amplitude
    # Fixed rate for stability and convexity
    f_b = p4 * np.exp(-0.5 * b)
    
    # 6. Interaction: Substrate * Temperature deviation
    # p5 is the coefficient
    f_st = p5 * f_s * (temp - 37.0)
    
    # 7. Interaction: Substrate * pH deviation
    # p6 is the coefficient
    f_sp = p6 * f_s * (pH - 7.0)
    
    # 8. Interaction: Temperature * pH deviation
    # p7 is the coefficient
    f_tp = p7 * (temp - 37.0) * (pH - 7.0)
    
    # 9. Linear term for substrate (to capture any non-saturation effects)
    # p8 is the coefficient
    f_s_lin = p8 * s
    
    # 10. Linear term for temperature (to capture any non-Gaussian effects)
    # p9 is the coefficient
    f_temp_lin = p9 * (temp - 37.0)
    
    # Combine additively
    y = p0 + p1 * f_s + f_temp + f_ph + f_b + f_st + f_sp + f_tp + f_s_lin + f_temp_lin
    
    return y
# EVOLVE-BLOCK-END
