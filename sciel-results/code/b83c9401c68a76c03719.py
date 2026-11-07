import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Tanh-based smooth elastic-plastic transition
    sigma = E(T)*eps * (1 - S(T) * tanh(W(T)*eps))
    Where S is the saturation factor (plateau reduction) and W is the transition width.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # Elastic Modulus E(T) = p0 + p1*T
    E = p0 + p1 * temp
    
    # Saturation Factor S(T) = p2 + p3*T
    # This determines how much the linear response is reduced at high strain.
    # Ideally S < 1 for physical behavior (stress doesn't drop to zero).
    S = p2 + p3 * temp
    
    # Transition Width W(T) = p4 + p5*T
    # Controls the steepness of the transition from linear to plateau.
    # Larger W means sharper knee.
    W = p4 + p5 * temp
    
    # Additional offset or shift for the transition center?
    # Let's keep it centered at 0 for simplicity, or add a small shift.
    # Let's use remaining params for a secondary term or refinement.
    # Maybe a small quadratic correction to the plateau?
    # Or just keep the core tanh model and use remaining params for temperature dependence of W and S?
    # Wait, I used 6 params. I have 4 left.
    # Let's refine:
    # E(T) = p0 + p1*T
    # S(T) = p2 + p3*T
    # W(T) = p4 + p5*T
    # Let's add a small linear correction term that is always active?
    # Or maybe a shift in strain: eps_shifted = eps - p6?
    # Let's try: sigma = E(T)*eps * (1 - S(T) * tanh(W(T)*(eps - p6))) + p7*eps + p8*eps^2 + p9*eps^3?
    # That might be too complex. Let's stick to the tanh saturation but make it slightly more flexible.
    
    # Let's use the remaining params to modulate the shape slightly.
    # Perhaps: sigma = E(T)*eps * (1 - S(T) * tanh(W(T)*eps)) + (p6 + p7*temp) * eps^2
    # This adds a quadratic term to account for initial curvature if needed.
    
    eps = strain
    eps_shifted = eps - p6  # p6 is the strain offset for the transition center
    
    # Ensure W is not zero to avoid division issues in tanh derivative? tanh(0)=0, fine.
    # But if W is very small, it's almost linear.
    
    tanh_term = np.tanh(W * eps_shifted)
    
    # The factor (1 - S * tanh) goes from 1 (at eps=0) to (1-S) (at large eps).
    # If S is close to 1, it saturates to 0. If S is small, it stays near linear.
    # For aluminum, stress doesn't drop to 0. So S should be < 1.
    # Let's constrain S implicitly? No, optimizer will find best values.
    
    linear_part = E * eps
    saturation_factor = 1.0 - S * tanh_term
    
    stress = linear_part * saturation_factor
    
    # Add a small quadratic correction for better fit in the knee region?
    # Let's use p7, p8, p9 for a small polynomial correction.
    # Correction = (p7 + p8*temp) * eps^2 + p9 * eps^3
    eps2 = eps * eps
    correction = (p7 + p8 * temp) * eps2 + p9 * eps2 * eps
    
    return stress + correction
