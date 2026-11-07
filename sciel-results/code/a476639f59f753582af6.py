import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center temperature for better conditioning
    T = temp - 0.5
    
    # Voce-like saturation part
    # sigma_inf(T) = p0 + p1*T + p2*T^2
    sigma_inf = p0 + p1*T + p2*T**2
    # k(T) = p3 + p4*T + p5*T^2
    k = p3 + p4*T + p5*T**2
    
    # Ensure k is positive to avoid numerical issues with exp
    k_safe = np.maximum(k, 1e-6)
    
    # Base Voce stress
    # sigma_voce = sigma_inf * (1 - exp(-k * strain))
    # Note: strain can be negative. exp handles it fine.
    # However, physically stress should be 0 at strain 0.
    # Voce form above satisfies this.
    sigma_voce = sigma_inf * (1.0 - np.exp(-k_safe * strain))
    
    # Polynomial correction terms (centered strain)
    # This allows the model to fit local variations better
    eps_c = strain - 0.5
    poly_correction = p6 * eps_c + p7 * eps_c**2 + p8 * eps_c**3 + p9 * eps_c**4
    
    return sigma_voce + poly_correction
