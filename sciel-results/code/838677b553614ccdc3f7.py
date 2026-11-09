"""Symbolic regression: stress-strain of an aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    
    Uses a combination of:
    - Elastic linear term (strain)
    - Nonlinear hardening terms (rational saturation)
    - Temperature modulation of yield and hardening
    
    params: array of 10 fitted constants
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent yield stress (decreases with T)
    # sigma_y(T) = p0 + p1 * T
    sigma_y = p0 + p1 * T
    
    # Temperature-dependent hardening modulus
    # E_h(T) = p2 + p3 * T
    E_h = p2 + p3 * T
    
    # Saturation stress (ultimate)
    # sigma_u(T) = p4 + p5 * T
    sigma_u = p4 + p5 * T
    
    # Characteristic strain for saturation
    eps_sat = p6 + p7 * T
    
    # Ensure eps_sat is positive for numerical stability
    eps_sat_safe = np.where(np.abs(eps_sat) < 1e-10, 1e-10, eps_sat)
    
    # Nonlinear hardening: smooth saturation using rational function
    # f(eps) = 1 - exp(-eps / eps_sat) approximated by rational: eps / (eps + eps_sat)
    # This gives smooth, differentiable behavior
    hardening_factor = eps / (eps + eps_sat_safe)
    
    # Elastic contribution for small strains
    # Use a smooth transition: for eps < yield_strain, elastic; above, plastic
    # yield_strain(T) = sigma_y / E where E ~ p8
    E = p8
    E_safe = np.where(np.abs(E) < 1e-10, 1e-10, E)
    
    yield_strain = sigma_y / E_safe
    
    # Smooth elastic-plastic transition using a smoothstep-like function
    # For eps < yield_strain: elastic (E * eps)
    # For eps > yield_strain: yield + hardening
    # Use smooth transition with parameter p9 controlling sharpness
    
    # Compute elastic stress
    stress_elastic = E_safe * eps
    
    # Compute plastic stress: sigma_y + (sigma_u - sigma_y) * hardening_factor
    stress_plastic = sigma_y + (sigma_u - sigma_y) * hardening_factor
    
    # Smooth blend between elastic and plastic
    # t = (eps - yield_strain) / (2 * yield_strain * p9)
    # Use tanh for smoothness
    transition_width = np.where(np.abs(p9) < 1e-10, 1e-10, p9)
    t = (eps - yield_strain) / (2.0 * yield_strain * transition_width)
    t_clipped = np.clip(t, -20, 20)  # avoid overflow in tanh
    blend = 0.5 * (1.0 + np.tanh(t_clipped))
    
    # Combine elastic and plastic
    stress = (1.0 - blend) * stress_elastic + blend * stress_plastic
    
    return stress
# EVOLVE-BLOCK-END
