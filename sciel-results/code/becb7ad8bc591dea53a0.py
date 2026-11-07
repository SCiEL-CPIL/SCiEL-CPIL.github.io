import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear stress-strain model for aluminium rod using a smooth
    hinge to separate elastic and plastic regimes, with temperature-dependent
    coefficients and polynomial corrections.
    
    Core: smooth hinge s*(1-exp(-k*s)) creates an explicit yield threshold,
    allowing independent fitting of elastic slope E and yield plateau Y.
    
    Refinement: low-order polynomial corrections absorb systematic deviations.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    # Elastic modulus
    E = p0 + p1 * temp
    # Yield plateau stress
    Y = p2 + p3 * temp
    # Hinge sharpness (kept positive for stability)
    k = np.abs(p4) + 1e-6
    # Post-yield hardening slope
    K = p5 + p6 * temp
    
    # Smooth hinge function: s * (1 - exp(-k*s))
    # At small s: ~ k*s^2 (quadratic, but scaled by E below)
    # At large s: ~ s (linear, representing plastic flow)
    # We use this to create a transition from elastic to plastic
    
    # Elastic part: linear in strain
    elastic = E * strain
    
    # Plastic part: activated after yield threshold
    # The hinge creates a smooth transition
    # We want: stress = elastic + plastic_correction
    # where plastic_correction becomes significant when strain exceeds yield threshold
    
    # Yield threshold in strain: e_y = Y / E (when elastic stress reaches yield)
    # But we use the smooth hinge to avoid explicit division
    
    # Smooth hinge: h(s) = s * (1 - exp(-k*s))
    # This is 0 at s=0, ~k*s^2 for small s, ~s for large s
    # We scale it to create the plastic regime
    
    # Use strain as the argument to the hinge
    # The hinge should activate when elastic stress approaches yield
    # So we normalize strain by the yield strain: s = strain * E / Y
    
    # Protect against division by zero
    Y_safe = np.where(np.abs(Y) < 1e-10, np.sign(Y) * 1e-10, Y)
    
    # Normalized strain for hinge activation
    s = strain * E / Y_safe
    
    # Smooth hinge: h(s) = s * (1 - exp(-k*s))
    # For negative s, we want the hinge to be 0 (no plastic flow in compression
    # for this model, or we handle it separately)
    # Actually, let's make it work for both signs:
    # h(s) = s * (1 - exp(-k*|s|)) for s > 0
    # h(s) = 0 for s <= 0 (or we can make it symmetric)
    
    # For simplicity and physical motivation, let's use:
    # plastic = Y * h(s) where h(s) = max(s, 0) * (1 - exp(-k*max(s, 0)))
    # This ensures plastic flow only when s > 0 (tension beyond yield)
    
    s_pos = np.maximum(s, 0.0)
    h = s_pos * (1.0 - np.exp(-k * s_pos))
    
    # Plastic contribution
    plastic = Y * h
    
    # Post-yield hardening: linear in the plastic strain
    # The plastic strain is approximately h (since h ~ s for large s)
    hardening = K * h
    
    # Total stress
    stress = elastic + plastic + hardening
    
    # Polynomial corrections for systematic deviations
    stress += (
        p7 * strain**2
        + p8 * temp * strain
        + p9 * temp**2
    )
    
    return stress
