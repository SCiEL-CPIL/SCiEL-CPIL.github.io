"""E. coli growth rate model: RBF expansion with biological gating."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using RBF expansion with biological priors.
    
    Model: mu = base * monod(s) * gauss(temp) * gauss(pH) * dens(b) + rbf_term + bias
    
    Where rbf_term is a sum of 4 Radial Basis Functions centered at learned points,
    allowing local flexibility to capture sharp optima and non-quadratic interactions.
    
    Parameters (10):
    p0: C (max growth rate scale)
    p1: K_s (Monod half-saturation)
    p2: T_opt (optimal temperature)
    p3: sigma_T (temperature width)
    p4: pH_opt (optimal pH)
    p5: sigma_pH (pH width)
    p6: b_half (density half-max inhibition)
    p7: RBF center 1 (temperature)
    p8: RBF center 2 (pH)
    p9: RBF width (shared)
    """
    # Deadline guard: return a safe default if we're running out of time
    # (This is a safeguard; the actual computation is fast)
    
    C = params[0]
    K_s = np.abs(params[1]) + 1e-6
    T_opt = params[2]
    sigma_T = np.abs(params[3]) + 0.1
    pH_opt = params[4]
    sigma_pH = np.abs(params[5]) + 0.1
    b_half = np.abs(params[6]) + 1e-6
    rbf_center_T = params[7]
    rbf_center_pH = params[8]
    rbf_width = np.abs(params[9]) + 0.1
    
    # Biological base model
    monod = s / (K_s + s + 1e-12)
    temp_term = np.exp(-0.5 * ((temp - T_opt) / sigma_T) ** 2)
    ph_term = np.exp(-0.5 * ((pH - pH_opt) / sigma_pH) ** 2)
    dens_term = 1.0 / (1.0 + (b / b_half) ** 2)
    
    base_model = C * monod * temp_term * ph_term * dens_term
    
    # RBF expansion for local flexibility
    # 4 RBFs centered at different points in (temp, pH) space
    # Centers are derived from the single RBF center params to stay within 10 total params
    # We create a small grid of centers around the main optimum
    
    # RBF 1: centered at (rbf_center_T, rbf_center_pH)
    dist1 = (temp - rbf_center_T) ** 2 + (pH - rbf_center_pH) ** 2
    rbf1 = np.exp(-dist1 / (2 * rbf_width ** 2))
    
    # RBF 2: shifted in temperature
    rbf_center_T2 = rbf_center_T + 0.5
    dist2 = (temp - rbf_center_T2) ** 2 + (pH - rbf_center_pH) ** 2
    rbf2 = np.exp(-dist2 / (2 * rbf_width ** 2))
    
    # RBF 3: shifted in pH
    rbf_center_pH2 = rbf_center_pH + 0.5
    dist3 = (temp - rbf_center_T) ** 2 + (pH - rbf_center_pH2) ** 2
    rbf3 = np.exp(-dist3 / (2 * rbf_width ** 2))
    
    # RBF 4: shifted in both
    dist4 = (temp - rbf_center_T2) ** 2 + (pH - rbf_center_pH2) ** 2
    rbf4 = np.exp(-dist4 / (2 * rbf_width ** 2))
    
    # The RBF coefficients are implicitly learned through the optimizer's ability
    # to adjust the centers and width. To give the optimizer more control, we use
    # the sign and magnitude of some existing params to modulate RBF contributions.
    # Here, we use a simple sum with unit weights, but scaled by C to maintain consistency.
    rbf_term = C * 0.1 * (rbf1 + rbf2 + rbf3 + rbf4)
    
    # Combine
    result = base_model + rbf_term
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
