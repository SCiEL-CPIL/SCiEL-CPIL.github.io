import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
        
        # Normalize inputs to roughly [-1, 1] to help BFGS
        b_norm = (b - 3.0) / 2.0
        s_norm = (s - 3.0) / 2.0
        temp_norm = (temp - 20.0) / 20.0
        ph_norm = (pH - 7.0) / 5.0
        
        # Monod term using normalized s? 
        # Monod is usually on raw s. Let's use raw s for Monod to keep biological meaning, 
        # but the quadratic part on normalized inputs.
        # Actually, if we use raw s in Monod, p1 needs to be around 1-5. Starting at 1 is fine.
        monod = s / (p1 + s)
        
        # Quadratic surface on normalized inputs
        quad = (p2 + p3 * b_norm + p4 * temp_norm + p5 * ph_norm + 
                p6 * b_norm**2 + p7 * temp_norm**2 + p8 * ph_norm**2 + 
                p9 * b_norm * temp_norm)
        
        return p0 * monod * quad
