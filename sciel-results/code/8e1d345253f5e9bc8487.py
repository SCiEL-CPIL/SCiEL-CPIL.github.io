import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a rational function model to capture elastic-plastic behavior.
    
    Model:
    sigma = (E(T) * eps + H(T) * eps^2) / (1 + L(T) * eps) + C1 * eps + C0
    
    Where:
    E(T) = p[0] + p[1] * T  (Young's Modulus)
    L(T) = p[2] + p[3] * T  (Yield Strain Scale)
    H(T) = p[4] + p[5] * T  (Hardening Coefficient)
    M(T) = p[6] + p[7] * T  (Hardening Saturation Scale) - Wait, I only have 10 params.
    
    Let's re-allocate:
    Term 1 (Elastic/Plastic transition):
    Num1 = (p[0] + p[1]*T) * eps
    Den1 = 1 + (p[2] + p[3]*T) * eps
    
    Term 2 (Additional Hardening):
    Num2 = (p[4] + p[5]*T) * eps^2
    Den2 = 1 + (p[6] + p[7]*T) * eps
    
    Term 3 (Linear Correction):
    p[8] * eps
    
    Term 4 (Offset):
    p[9]
    
    Total = Term1/Den1 + Term2/Den2 + Term3 + Term4
    """
    p = params
    eps = strain
    T = temp
    
    # Temperature dependent coefficients
    E_T = p[0] + p[1] * T
    L_T = p[2] + p[3] * T
    H_T = p[4] + p[5] * T
    M_T = p[6] + p[7] * T
    
    # Calculate denominators
    den1 = 1.0 + L_T * eps
    den2 = 1.0 + M_T * eps
    
    # Safe division to avoid NaN/Inf during optimization steps
    # If denominator is very small, we treat the term as 0 to keep finite
    # This is a heuristic to prevent crash, though physically inaccurate at singularity
    # However, BFGS should avoid these regions if MSE is high.
    
    # Using np.divide with where clause
    term1_num = E_T * eps
    term1 = np.divide(term1_num, den1, out=np.zeros_like(term1_num), where=np.abs(den1) > 1e-6)
    
    term2_num = H_T * eps**2
    term2 = np.divide(term2_num, den2, out=np.zeros_like(term2_num), where=np.abs(den2) > 1e-6)
    
    term3 = p[8] * eps
    term4 = p[9]
    
    return term1 + term2 + term3 + term4
