import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Basis terms:
    # 0: constant
    # 1: x
    # 2: y
    # 3: z
    # 4: x^2
    # 5: y^2
    # 6: z^2
    # 7: x*y
    # 8: y*z
    # 9: z*x
    
    term_0 = np.ones_like(x)
    term_1 = x
    term_2 = y
    term_3 = z
    term_4 = x * x
    term_5 = y * y
    term_6 = z * z
    term_7 = x * y
    term_8 = y * z
    term_9 = z * x
    
    return (p[0] * term_0 + 
            p[1] * term_1 + 
            p[2] * term_2 + 
            p[3] * term_3 + 
            p[4] * term_4 + 
            p[5] * term_5 + 
            p[6] * term_6 + 
            p[7] * term_7 + 
            p[8] * term_8 + 
            p[9] * term_9)
