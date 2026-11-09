import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a linear combination of biological basis functions.
    The model is linear in the 10 parameters, ensuring a convex loss landscape for BFGS.
    
    Basis functions:
    1. Bias
    2. ln(1 + b)       : Substrate/Density saturation
    3. ln(1 + s)       : Substrate saturation
    4. temp            : Linear temperature effect
    5. pH              : Linear pH effect
    6. (temp - 37)^2   : Quadratic temperature deviation (optimum around 37C)
    7. (pH - 7)^2      : Quadratic pH deviation (optimum around 7)
    8. ln(1 + b) * ln(1 + s) : Interaction between density and substrate
    9. temp * pH       : Interaction between temperature and pH
    10. b * s          : Interaction between density and substrate
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute basis functions
    # Use np.log1p for numerical stability and accuracy for small arguments
    log_b = np.log1p(b)
    log_s = np.log1p(s)
    
    temp_dev_sq = (temp - 37.0) ** 2
    ph_dev_sq = (pH - 7.0) ** 2
    
    # Construct the linear combination
    y = (p0 
         + p1 * log_b 
         + p2 * log_s 
         + p3 * temp 
         + p4 * pH 
         + p5 * temp_dev_sq 
         + p6 * ph_dev_sq 
         + p7 * log_b * log_s 
         + p8 * temp * pH 
         + p9 * b * s)
         
    return y
