import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Structural change: Use normalized Chebyshev polynomials of the first kind.
        Normalization ensures that each basis function has unit L2 norm on the
        domain, which improves the conditioning of the Hessian for BFGS and
        makes the parameter scales more uniform, aiding convergence.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Map x from [0.1, 6] to [-1, 1]
        x_min = 0.1
        x_max = 6.0
        t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
        
        # Compute Chebyshev polynomials T_0 ... T_9
        T0 = np.ones_like(t)
        T1 = t
        
        basis = np.zeros((len(t), 10))
        basis[:, 0] = T0
        basis[:, 1] = T1
        
        for k in range(2, 10):
            Tk = 2.0 * t * T1 - T0
            basis[:, k] = Tk
            T0, T1 = T1, Tk
            
        # Normalize each column to have unit L2 norm (approximated by RMS * sqrt(N))
        # Using sum of squares for robustness
        col_norms = np.sqrt(np.sum(basis**2, axis=0))
        # Avoid division by zero
        col_norms[col_norms == 0] = 1.0
        basis = basis / col_norms
        
        # Linear combination
        return basis @ params
