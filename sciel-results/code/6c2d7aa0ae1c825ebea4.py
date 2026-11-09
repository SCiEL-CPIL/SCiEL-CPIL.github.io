import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Unpack parameters
        E0, E1 = params[0], params[1]
        ey0, ey1 = params[2], params[3]
        H0, H1 = params[4], params[5]
        d0, d1 = params[6], params[7]
        K0, K1 = params[8], params[9]
        
        # Temperature-dependent coefficients
        E = E0 + E1 * temp
        ey = ey0 + ey1 * temp
        H = H0 + H1 * temp
        delta = d0 + d1 * temp
        K = K0 + K1 * temp
        
        # Ensure delta is positive to enforce necking after yield
        # Using softplus or just exp? Exp is safer for positivity but might constrain the scale.
        # Let's use np.maximum(0.01, delta) to avoid zero/negative gaps.
        delta_safe = np.maximum(0.01, delta)
        
        en = ey + delta_safe
        
        # Calculate terms
        elastic = E * strain
        hardening = H * np.maximum(0.0, strain - ey)
        necking = -K * np.maximum(0.0, strain - en)
        
        return elastic + hardening + necking
