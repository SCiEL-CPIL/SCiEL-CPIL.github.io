import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using multiplicative temperature-modulated elastic-plastic form.
    
    Uses:
    - Temperature-dependent elastic modulus (linear in T)
    - Temperature-dependent yield stress (linear in T)
    - Smooth sinh-based saturation for plastic hardening
    - Multiplicative temperature modulation of hardening
    """
    eps = strain
    T = temp
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * T
    
    # Temperature-dependent yield stress
    sigma_y = p2 + p3 * T
    
    # Smooth saturation using sinh to model elastic-plastic transition
    # sinh(x) ~ x for small x, grows exponentially for large x
    # We use tanh-like behavior: sinh(a*eps) / sinh(a*eps_max) or similar
    # Better: use (exp(a*eps) - 1) / (exp(a*eps_max) - 1) but that's not smooth at 0
    # Use: tanh(a * eps) which is smooth and saturates
    # Or: (1 - exp(-a*eps)) which is smooth and saturates
    # Let's use a combination: elastic part + plastic hardening
    
    # Elastic contribution: E * eps for small strains
    elastic_part = E * eps
    
    # Plastic hardening: smooth saturation function
    # Using sinh-based: sinh(p4 * eps) / p4 approximates eps for small eps, grows for large
    # But we want it to saturate, so use tanh
    hardening_scale = p4
    hardening_shape = p5
    
    # Smooth yield transition: tanh(hardening_shape * eps) 
    # This is ~hardening_shape*eps for small eps, saturates at 1
    yield_transition = np.tanh(hardening_shape * eps)
    
    # Plastic strain contribution: sigma_y * (yield_transition - 1) 
    # This is 0 at eps=0, approaches -sigma_y as eps->inf
    # But we want positive hardening, so: sigma_y * (1 - yield_transition)
    # That gives sigma_y at eps=0, approaches 0 - not right
    
    # Better: elastic + plastic where plastic = sigma_y * (1 - exp(-hardening_scale * eps))
    # This is 0 at eps=0, approaches sigma_y
    plastic_part = sigma_y * (1.0 - np.exp(-hardening_scale * eps))
    
    # Temperature modulation of hardening
    temp_mod = p6 + p7 * T
    
    # Combined: elastic + temperature-modulated plastic
    stress = elastic_part + temp_mod * plastic_part
    
    # Add small correction terms for curvature
    correction = p8 * eps * eps + p9 * T * eps
    
    return stress + correction
