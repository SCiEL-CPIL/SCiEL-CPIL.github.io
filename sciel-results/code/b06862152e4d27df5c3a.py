import numpy as np

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate model using a Generalized Additive Model (GAM).
    
    Strategy:
    Instead of a global quadratic surface with interactions, we use a sum of 
    univariate piecewise-linear basis functions (hats) for each input.
    This decouples the complexity, making the loss landscape smoother and 
    easier for BFGS to navigate. The 10 parameters are the coefficients 
    for these basis functions.
    
    Inputs:
    b: Population density [1, 5]
    s: Substrate concentration [1, 5]
    temp: Temperature [1, 40]
    pH: pH level [2, 12]
    
    Basis Functions:
    We define 3 hat functions per variable (centers at 25%, 50%, 75% of range).
    Total 12 basis functions. We use 10 parameters, mapping them to:
    p[0]: Intercept (bias)
    p[1]-p[9]: Coefficients for 9 selected hat functions to cover the space.
    
    Specifically:
    b: hats at 2, 3, 4 (params 1, 2, 3)
    s: hats at 2, 3, 4 (params 4, 5, 6)
    temp: hats at 10, 20, 30 (params 7, 8, 9)
    pH: We omit specific pH hats in this 10-param budget to prioritize 
    the other variables, or we could merge. 
    Actually, let's distribute:
    b: 2 hats (params 1, 2) at 2, 4
    s: 2 hats (params 3, 4) at 2, 4
    temp: 3 hats (params 5, 6, 7) at 10, 20, 30
    pH: 2 hats (params 8, 9) at 4, 8
    p[0]: Intercept
    
    This covers all 4 inputs with non-linear flexibility.
    """
    p = params
    
    # Define centers for hat functions based on input ranges
    # b: [1, 5] -> centers 2.0, 4.0
    # s: [1, 5] -> centers 2.0, 4.0
    # temp: [1, 40] -> centers 10.0, 20.0, 30.0
    # pH: [2, 12] -> centers 4.0, 8.0
    
    # Helper to compute hat function: max(0, 1 - |x - c| / w)
    # We use width w=1.0 for b,s,pH and w=5.0 for temp to match scale
    
    def hat(x, c, w):
        return np.maximum(0.0, 1.0 - np.abs(x - c) / w)
    
    # Compute basis values
    # b basis
    b_hat1 = hat(b, 2.0, 1.5)
    b_hat2 = hat(b, 4.0, 1.5)
    
    # s basis
    s_hat1 = hat(s, 2.0, 1.5)
    s_hat2 = hat(s, 4.0, 1.5)
    
    # temp basis
    t_hat1 = hat(temp, 10.0, 5.0)
    t_hat2 = hat(temp, 20.0, 5.0)
    t_hat3 = hat(temp, 30.0, 5.0)
    
    # pH basis
    p_hat1 = hat(pH, 4.0, 2.0)
    p_hat2 = hat(pH, 8.0, 2.0)
    
    # Linear combination
    # p[0] is intercept
    # p[1], p[2] for b
    # p[3], p[4] for s
    # p[5], p[6], p[7] for temp
    # p[8], p[9] for pH
    
    z = (p[0] + 
         p[1] * b_hat1 + p[2] * b_hat2 +
         p[3] * s_hat1 + p[4] * s_hat2 +
         p[5] * t_hat1 + p[6] * t_hat2 + p[7] * t_hat3 +
         p[8] * p_hat1 + p[9] * p_hat2)
    
    # Apply softplus to ensure non-negative growth rate and smoothness
    # This is crucial for biological validity and optimization stability
    return np.log1p(np.exp(z))
