import numpy as np

def _cheb_t0(t):
    return np.ones_like(t)

def _cheb_t1(t):
    return t

def _cheb_t2(t):
    return 2.0 * t**2 - 1.0

def _cheb_t3(t):
    return 4.0 * t**3 - 3.0 * t

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Separable Chebyshev product basis: T_i(x) * T_j(y)
    # Degrees: x in {0,1,2,3}, y in {0,1,2} -> 4*3 = 12, but we need exactly 10 params
    # Use: T_0*T_0, T_1*T_0, T_2*T_0, T_3*T_0, T_0*T_1, T_1*T_1, T_2*T_1, T_3*T_1, T_0*T_2, T_1*T_2
    # That's 10 terms
    
    Tx0 = _cheb_t0(x)
    Tx1 = _cheb_t1(x)
    Tx2 = _cheb_t2(x)
    Tx3 = _cheb_t3(x)
    
    Ty0 = _cheb_t0(y)
    Ty1 = _cheb_t1(y)
    Ty2 = _cheb_t2(y)
    
    pred = (params[0] * Tx0 * Ty0 +
            params[1] * Tx1 * Ty0 +
            params[2] * Tx2 * Ty0 +
            params[3] * Tx3 * Ty0 +
            params[4] * Tx0 * Ty1 +
            params[5] * Tx1 * Ty1 +
            params[6] * Tx2 * Ty1 +
            params[7] * Tx3 * Ty1 +
            params[8] * Tx0 * Ty2 +
            params[9] * Tx1 * Ty2)
    
    return pred
