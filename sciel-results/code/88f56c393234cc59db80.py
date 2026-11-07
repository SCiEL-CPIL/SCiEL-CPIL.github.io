import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a quadratic polynomial in substrate, temperature, and pH,
    plus a linear term for population density.
    
    Inputs are normalized to [0, 1] to improve numerical stability.
    params: [c0, cs, cs2, cT, cT2, cp, cp2, cST, cSP, cb]
    """
    # Unpack parameters
    c0 = params[0]
    cs = params[1]
    cs2 = params[2]
    cT = params[3]
    cT2 = params[4]
    cp = params[5]
    cp2 = params[6]
    cST = params[7]
    cSP = params[8]
    cb = params[9]
    
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 4.999] -> approx [1, 5]
    # s: [1, 4.999] -> approx [1, 5]
    # temp: [1.002, 39.98] -> approx [1, 40]
    # pH: [2, 12]
    
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    T_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Construct basis functions
    # 1. Constant
    # 2. Linear s
    # 3. Quadratic s
    # 4. Linear T
    # 5. Quadratic T
    # 6. Linear p
    # 7. Quadratic p
    # 8. Interaction s*T
    # 9. Interaction s*p
    # 10. Linear b
    
    y = c0 \
        + cs * s_n \
        + cs2 * s_n**2 \
        + cT * T_n \
        + cT2 * T_n**2 \
        + cp * p_n \
        + cp2 * p_n**2 \
        + cST * s_n * T_n \
        + cSP * s_n * p_n \
        + cb * b_n
        
    return y
