"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a mixed basis of polynomial, trigonometric,
    exponential, and rational terms. This approach avoids the ill-conditioning of
    high-degree polynomials and covers a wider variety of potential target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Define a diverse and well-conditioned basis of 10 functions:
    # 0: Constant
    # 1: Linear
    # 2: Quadratic
    # 3: Sine
    # 4: Cosine
    # 5: Sine(2x)
    # 6: Cosine(2x)
    # 7: Exponential
    # 8: Reciprocal
    # 9: Square root (or another low-order term, let's use x^0.5 or log(x)? 
    #    Log(x) is undefined at 0, but x>=0.1 so it's safe. 
    #    Let's use log(x) as it's common in physics/chem.
    #    Actually, let's stick to smooth functions. x^0.5 is smooth on [0.1, 6].
    #    Let's use: 1, x, x^2, sin(x), cos(x), sin(2x), cos(2x), exp(x), 1/x, log(x)
    
    # Note: log(x) can be sensitive if x is close to 0, but min x is 0.1, so log(0.1) ~ -2.3. Safe.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = np.sin(x)
    b4 = np.cos(x)
    b5 = np.sin(2*x)
    b6 = np.cos(2*x)
    b7 = np.exp(x)
    b8 = 1.0 / x
    b9 = np.log(x)
    
    # Stack basis functions into a matrix of shape (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Prediction is basis @ params
    return basis @ params
# EVOLVE-BLOCK-END
