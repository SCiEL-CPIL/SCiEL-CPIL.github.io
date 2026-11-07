import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible polynomial response surface for E. coli growth rate.
    
    This model uses a 4-variable quadratic polynomial (degree 2) to capture
    arbitrary interactions and non-linearities. Unlike the previous model which
    forced specific functional forms (Monod, Gaussian), this allows the optimizer
    to discover the actual shape of the response surface, including potential
    non-monotonic behaviors or complex interactions between variables.
    
    Parameters (10 total):
    p0: Intercept
    p1-p4: Linear coefficients for b, s, temp, pH
    p5-p7: Quadratic coefficients for b^2, s^2, temp^2
    p8: Cross-term coefficient for b*s (often significant in growth)
    p9: Cross-term coefficient for temp*pH (environmental interaction)
    
    Note: We omit some cross terms (b^2 is included, s^2, temp^2 are included) 
    to fit within 10 params while covering the most likely dominant non-linearities.
    A full quadratic would need 1 + 4 + 6 = 11 terms. We prioritize:
    - Linear (4)
    - Pure Quadratic (3: b, s, temp - pH quadratic is less critical or absorbed)
    - Key Interactions (2: b*s, temp*pH)
    - Intercept (1)
    Total: 10.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    b = np.asarray(b)
    s = np.asarray(s)
    temp = np.asarray(temp)
    pH = np.asarray(pH)
    
    # Construct polynomial terms
    # Linear
    term_b = p1 * b
    term_s = p2 * s
    term_t = p3 * temp
    term_ph = p4 * pH
    
    # Quadratic (Pure)
    term_b2 = p5 * (b ** 2)
    term_s2 = p6 * (s ** 2)
    term_t2 = p7 * (temp ** 2)
    
    # Interactions
    term_bs = p8 * (b * s)
    term_tp = p9 * (temp * pH)
    
    # Sum
    return p0 + term_b + term_s + term_t + term_ph + term_b2 + term_s2 + term_t2 + term_bs + term_tp
