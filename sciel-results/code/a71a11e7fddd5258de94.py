import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2t T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    T6 = 2 * t * T5 - T4
    T7 = 2 * t * T6 - T5
    T8 = 2 * t * T7 - T6
    T9 = 2 * t * T8 - T7
    
    basis = np.column_stack([
        T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
    ])
    
    return basis @ params
