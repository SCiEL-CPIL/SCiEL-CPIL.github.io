import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Nonlinear parametric model:
    # The parameters p define the structure of the function, not just linear coefficients.
    # This allows the model to capture functions that are not linear combinations of a fixed basis.
    
    # Unpack parameters
    # p[0]: Amplitude for sine term
    # p[1]: Frequency factor for x in sine term
    # p[2]: Frequency factor for y in sine term
    # p[3]: Phase shift for sine term
    # p[4]: Amplitude for cosine term
    # p[5]: Frequency factor for x in cosine term
    # p[6]: Frequency factor for y in cosine term
    # p[7]: Phase shift for cosine term
    # p[8]: Constant offset
    # p[9]: Quadratic amplitude (to capture potential parabolic trends)
    
    A1 = params[0]
    f1x = params[1]
    f1y = params[2]
    phi1 = params[3]
    
    A2 = params[4]
    f2x = params[5]
    f2y = params[6]
    phi2 = params[7]
    
    C = params[8]
    Q = params[9]
    
    # Compute arguments
    arg1 = f1x * x + f1y * y + phi1
    arg2 = f2x * x + f2y * y + phi2
    
    # Compute function values
    # Using sin and cos with variable frequencies and phases
    # The quadratic term Q*(x^2 + y^2) helps capture radial symmetry if present
    val = A1 * np.sin(arg1) + A2 * np.cos(arg2) + C + Q * (x**2 + y**2)
    
    return val
