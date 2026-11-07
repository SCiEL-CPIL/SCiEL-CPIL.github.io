import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hyperbolic-sine hardening model for stress-strain of an aluminium rod.
    
    sigma = A(T) * sinh(B(T) * strain) / B(T) + C(T) * strain + D(T)
    
    Where:
    A(T) = p0 + p1*T + p2*T^2     (Amplitude / elastic modulus scale)
    B(T) = p3 + p4*T + p5*T^2     (Hardening rate / inverse characteristic strain)
    C(T) = p6 + p7*T              (Linear correction / thermal stress)
    D(T) = p8 + p9*T              (Offset / pre-stress)
    
    The sinh model naturally transitions from linear (elastic) to saturating (plastic)
    behavior as strain increases, capturing both regimes smoothly.
    """
    p = params
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    eps = strain
    
    # Temperature-dependent parameters
    A = p[0] + p[1] * T + p[2] * T * T
    B = p[3] + p[4] * T + p[5] * T * T
    C = p[6] + p[7] * T
    D = p[8] + p[9] * T
    
    # Ensure B is not too close to zero to avoid numerical issues with sinh(B*x)/B
    B_safe = np.where(np.abs(B) < 1e-8, 1e-8 * np.sign(B + 1e-12), B)
    
    # Compute sinh(B*eps) / B with clipping to avoid overflow
    arg = B_safe * eps
    arg_clipped = np.clip(arg, -50, 50)
    
    # sinh(x) for small x behaves like x, so sinh(B*eps)/B ~ eps for small B*eps
    # This gives us the linear elastic regime naturally
    sinh_arg = np.sinh(arg_clipped)
    core_term = sinh_arg / B_safe
    
    # Final stress
    sigma = A * core_term + C * eps + D
    
    return sigma
