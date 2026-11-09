import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of two sinusoidal functions with optimizable frequencies, phases, and amplitudes.
    This structural change moves to a periodic/non-polynomial basis which can 
    capture oscillatory or smooth non-linear behaviors that polynomials and 
    fixed RBFs might miss or fit poorly.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # First sine term: p8 * sin(p0*x + p1*y + p2*z + p3)
    arg1 = p0 * x + p1 * y + p2 * z + p3
    term1 = p8 * np.sin(arg1)
    
    # Second sine term: p9 * sin(p4*x + p5*y + p6*z + p7)
    arg2 = p4 * x + p5 * y + p6 * z + p7
    term2 = p9 * np.sin(arg2)
    
    pred = term1 + term2
    
    return pred
