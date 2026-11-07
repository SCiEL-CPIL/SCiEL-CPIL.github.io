import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a hybrid model:
    - Physical terms (Duffing, Van der Pol, driving)
    - Gaussian RBF terms for localized nonlinear corrections
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Physical terms
    term0 = p0
    term1 = p1 * x
    term2 = p2 * (x ** 3)
    term3 = p3 * v
    term4 = p4 * (v ** 3)
    term5 = p5 * x * v
    term6 = p6 * np.sin(t)
    term7 = p7 * np.cos(t)
    term8 = p8 * t
    term9 = p9 * (x ** 2)
    
    result = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
    return result
