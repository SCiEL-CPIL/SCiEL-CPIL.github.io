"""Piecewise elastic-plastic model with temperature-dependent saturation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = E(T)*strain (elastic) + H(T)*max(0, strain - eps_y(T)) (plastic hardening)
             - S(T)*exp(-strain/eps_s(T)) (saturation/softening correction)
    
    This structure separates the initial linear response from the hardening regime,
    which is often easier for BFGS to fit than a single complex exponential.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Elastic Modulus E(T) = p0 + p1*T
    E = p0 + p1 * temp
    
    # 2. Yield Strain eps_y(T) = p2 + p3*T
    # Clip to ensure it stays within a reasonable physical range to avoid numerical issues
    # though BFGS should handle it. We use a soft clip via tanh-like behavior or just direct use.
    # Let's keep it direct for simplicity, but ensure it's not zero to avoid division issues if used in denominators (it isn't here).
    eps_y = p2 + p3 * temp
    
    # 3. Plastic Hardening Modulus H(T) = p4 + p5*T
    H = p4 + p5 * temp
    
    # 4. Saturation Stress S(T) = p6 + p7*T
    S = p6 + p7 * temp
    
    # 5. Saturation Strain Scale eps_s(T) = p8 + p9*T
    # Ensure eps_s is positive for the exponential decay to be stable
    # If p8+p9*T becomes negative, exp(-strain/eps_s) blows up for positive strain.
    # We can enforce positivity by adding a small constant or using abs, but abs breaks differentiability.
    # Better: constrain via the model structure? No, we just fit.
    # To be safe against BFGS wandering into negative eps_s, we can square it or add a constant.
    # Let's use (p8 + p9*T)**2 + 1e-6 to guarantee positivity and differentiability.
    eps_s = (p8 + p9 * temp)**2 + 1e-6
    
    # Elastic component
    stress_elastic = E * strain
    
    # Plastic component: Only active if strain > eps_y
    # max(0, strain - eps_y) is differentiable almost everywhere (gradient 0 or 1).
    # BFGS can handle this non-smoothness usually, but let's use a smooth approximation if needed?
    # Standard max is often fine.
    plastic_strain = np.maximum(0.0, strain - eps_y)
    stress_plastic = H * plastic_strain
    
    # Saturation term: Reduces stress as strain increases towards saturation
    # This term should be subtracted from the linear growth to create a plateau.
    # At strain=0, exp(0)=1, so term is S.
    # At strain->inf, exp(-inf)=0, so term is 0.
    # So stress = Elastic + Plastic - S*exp(-strain/eps_s)
    # Wait, if we subtract S*exp, at strain=0 we subtract S.
    # This might be better: Stress = Elastic + Plastic - S * (1 - exp(-strain/eps_s)) ?
    # No, that adds a hardening term.
    # To get a plateau, we want the slope to decrease.
    # Let's use the standard Voce form for the whole thing?
    # The current inspiration 1 (score 1.28) used Voce.
    # Let's try a "Bilinear + Exponential Decay" model.
    
    # Alternative: 
    # Stress = E*strain + H*max(0, strain-eps_y) - S*exp(-strain/eps_s)
    # This creates a dip at low strain if S is large.
    
    # Let's stick to a robust Bilinear model with a temperature-dependent yield point.
    # The "saturation" is often better modeled by the hardening modulus H itself decreasing?
    # Or just a simple Bilinear model:
    # Stress = E*strain + H*max(0, strain - eps_y)
    # This has 6 params (E, eps_y, H, and their temp coeffs). We have 10.
    # We can add a quadratic term for curvature in the plastic region.
    
    # Revised Model:
    # 1. Linear Elastic: (p0 + p1*T) * strain
    # 2. Plastic Hardening: (p2 + p3*T) * max(0, strain - (p4 + p5*T))
    # 3. Quadratic Correction: (p6 + p7*T) * (max(0, strain - (p4 + p5*T)))**2
    # 4. Constant Offset: p8 + p9*T
    
    eps_y = p4 + p5 * temp
    E = p0 + p1 * temp
    H_lin = p2 + p3 * temp
    H_quad = p6 + p7 * temp
    offset = p8 + p9 * temp
    
    plastic_strain = np.maximum(0.0, strain - eps_y)
    
    stress = E * strain + H_lin * plastic_strain + H_quad * plastic_strain**2 + offset
    
    return stress
# EVOLVE-BLOCK-END
