"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with an additive basis of biologically motivated terms.
    
    This approach combines the structural flexibility of an additive model (which is 
    easier for BFGS to optimize than a purely multiplicative one) with biologically 
    informed basis functions:
    - Monod saturation for substrate
    - Gaussian temperature optimum (fixed center 37, sigma 12)
    - Gaussian pH optimum (fixed center 7, sigma 3)
    - Linear and quadratic density terms
    - Key interaction terms between substrate, density, and environment
    
    Inputs are normalized or transformed to ensure well-conditioned optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # 1. Normalize/Transform Inputs
    # b range: [1, 5] -> center 3, scale 2
    b_n = (b - 3.0) / 2.0
    
    # s range: [1, 5] -> Monod transform s/(1+s) range [0.5, 0.833]
    # We use the raw Monod value, it's already bounded and meaningful.
    monod_s = s / (1.0 + s)
    
    # temp range: [1, 40]. E. coli optimum ~37C.
    # Gaussian: exp(-((temp - 37)^2) / (2 * 12^2))
    temp_diff = temp - 37.0
    temp_gauss = np.exp(-(temp_diff ** 2) / 288.0)  # 2 * 12^2 = 288
    
    # pH range: [2, 12]. E. coli optimum ~7.
    # Gaussian: exp(-((pH - 7)^2) / (2 * 3^2))
    ph_diff = pH - 7.0
    ph_gauss = np.exp(-(ph_diff ** 2) / 18.0)       # 2 * 3^2 = 18

    # 2. Construct Additive Basis
    # p0: Bias
    pred = np.full_like(b_n, p0, dtype=np.float64)
    
    # p1: Substrate saturation
    pred += p1 * monod_s
    
    # p2: Temperature effect
    pred += p2 * temp_gauss
    
    # p3: pH effect
    pred += p3 * ph_gauss
    
    # p4, p5: Density effects (linear and quadratic inhibition)
    pred += p4 * b_n
    pred += p5 * b_n * b_n
    
    # p6: Interaction Substrate * Temperature
    # Growth rate depends on both substrate availability and temperature
    pred += p6 * monod_s * temp_gauss
    
    # p7: Interaction Substrate * pH
    pred += p7 * monod_s * ph_gauss
    
    # p8: Interaction Density * Temperature
    # Inhibition might be stronger at optimal temp
    pred += p8 * b_n * temp_gauss
    
    # p9: Interaction Density * pH
    pred += p9 * b_n * ph_gauss

    return pred
# EVOLVE-BLOCK-END
