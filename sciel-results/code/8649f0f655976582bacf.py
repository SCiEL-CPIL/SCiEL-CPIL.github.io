import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters Chebyshev polynomial model.
    
    stress = sum_{d=0}^{4} (a_d + b_d * T) * T_d(S)
    
    where T_d is the d-th Chebyshev polynomial of the first kind,
    evaluated at S (strain). This is linear in the 10 parameters
    [a_0, b_0, a_1, b_1, a_2, b_2, a_3, b_3, a_4, b_4],
    so BFGS converges to the global optimum efficiently.
    """
    S = strain
    T = temp
    
    # Extract parameters: (a_d, b_d) for d = 0..4
    a0, b0 = params[0], params[1]
    a1, b1 = params[2], params[3]
    a2, b2 = params[4], params[5]
    a3, b3 = params[6], params[7]
    a4, b4 = params[8], params[9]
    
    # Compute Chebyshev polynomials T_0..T_4 of S
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    T0 = np.ones_like(S)
    T1 = S
    T2 = 2.0 * S * S - 1.0
    T3 = 4.0 * S * S * S - 3.0 * S
    T4 = 8.0 * S * S * S * S - 8.0 * S * S + 1.0
    
    # Temperature-dependent coefficients
    c0 = a0 + b0 * T
    c1 = a1 + b1 * T
    c2 = a2 + b2 * T
    c3 = a3 + b3 * T
    c4 = a4 + b4 * T
    
    # Linear combination
    result = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3 + c4 * T4
    
    return result
