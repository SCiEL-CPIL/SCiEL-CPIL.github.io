import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Map x to [-1, 1]
        t = x / 3.0
        
        # Pre-compute Chebyshev basis
        # T is a matrix where row i is T_i(x)
        # We can compute this efficiently
        
        # T_0 = 1
        # T_1 = t
        # T_n = 2t T_{n-1} - T_{n-2}
        
        N = len(x)
        T0 = np.ones(N)
        T1 = t.copy()
        
        # Initialize result with p0*T0 + p1*T1
        result = params[0] * T0 + params[1] * T1
        
        # Add higher order terms
        if N > 0:
            T_prev = T0
            T_curr = T1
            for i in range(2, 10):
                T_next = 2.0 * t * T_curr - T_prev
                result += params[i] * T_next
                T_prev = T_curr
                T_curr = T_next
                
        return result
