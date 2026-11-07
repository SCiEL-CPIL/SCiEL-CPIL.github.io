import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a hidden function using a linear combination of a diverse basis of functions.
    The 10 parameters are fitted by the evaluator using BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 basis functions
    # 0: Constant
    b0 = np.ones_like(x)
    # 1-5: Polynomials
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    # 6-9: Trigonometric
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2*x)
    b8 = np.cos(2*x)
    # 10: Gaussian bump
    b9 = np.exp(-x**2)
    
    # Combine with parameters
    # params is an array of length 10
    p = params
    
    return p[0]*b0 + p[1]*b1 + p[2]*b2 + p[3]*b3 + p[4]*b4 + \
           p[5]*b5 + p[6]*b6 + p[7]*b7 + p[8]*b8 + p[9]*b9
