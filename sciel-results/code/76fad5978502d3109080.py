"""Symbolic regression with 3 inputs – Chebyshev Polynomial Basis."""
import numpy as np

def _map_to_unit_interval(t):
    """Map t from [0.5, 2] to [-1, 1]."""
    # t_min = 0.5, t_max = 2.0
    # u = (2 * t - (t_max + t_min)) / (t_max - t_min)
    return (2.0 * t - 2.5) / 1.5

def _chebyshev_T1(u):
    """T_1(u) = u"""
    return u

def _chebyshev_T2(u):
    """T_2(u) = 2*u^2 - 1"""
    return 2.0 * u**2 - 1.0

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Chebyshev polynomials mapped to the input domain.
    
    Basis functions (10 total):
    1. 1 (Constant)
    2. T1(x)
    3. T1(y)
    4. T1(z)
    5. T2(x)
    6. T2(y)
    7. T2(z)
    8. T1(x) * T1(y)
    9. T1(x) * T1(z)
    10. T1(y) * T1(z)
    
    params: array of length 10 corresponding to the coefficients above.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map inputs to [-1, 1]
    ux = _map_to_unit_interval(x)
    uy = _map_to_unit_interval(y)
    uz = _map_to_unit_interval(z)
    
    # Compute Basis Functions
    b0 = np.ones_like(ux)
    b1 = ux
    b2 = uy
    b3 = uz
    b4 = _chebyshev_T2(ux)
    b5 = _chebyshev_T2(uy)
    b6 = _chebyshev_T2(uz)
    b7 = ux * uy
    b8 = ux * uz
    b9 = uy * uz
    
    # Stack into matrix (n, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    # Ensure params is shape (10,)
    if params.shape != (10,):
        params = params[:10]
        
    pred = basis @ params
    
    return pred
