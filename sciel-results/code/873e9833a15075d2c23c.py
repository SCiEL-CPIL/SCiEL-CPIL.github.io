"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a flexible additive + multiplicative hybrid.
    
    The model combines:
    1. A polynomial response in each input (degree 2) to capture non-linearities.
    2. A saturating term for substrate (Monod-like).
    3. A Gaussian-like penalty for temperature and pH.
    4. An inhibitory term for density (b).
    
    This structure is more flexible than a pure product form, allowing the optimizer
    to find additive corrections if the multiplicative assumption is too strict.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safety guards for parameters that appear in denominators or exponents
    # p1: Ks (half-saturation constant for substrate)
    Ks = np.maximum(np.abs(p1), 1e-6)
    
    # p2: Optimal temperature
    T_opt = p2
    
    # p3: Temperature width
    sigma_T = np.maximum(np.abs(p3), 1e-6)
    
    # p4: Optimal pH
    pH_opt = p4
    
    # p5: pH width
    sigma_pH = np.maximum(np.abs(p5), 1e-6)
    
    # p6: Inhibition constant for density
    K_b = np.maximum(np.abs(p6), 1e-6)
    
    # p7: Hill coefficient for density inhibition
    n_hill = np.maximum(np.abs(p7), 0.1)
    
    # p8: Scaling factor for substrate saturation term
    # We use a softplus-like or just a linear scale to avoid stiffness
    scale_s = p8
    
    # p9: Global scale
    global_scale = p9
    
    # 1. Substrate Term: Monod
    # s / (Ks + s)
    monod_s = s / (Ks + s)
    
    # 2. Temperature Term: Gaussian
    # exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
    temp_term = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T ** 2))
    
    # 3. pH Term: Gaussian
    # exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
    ph_term = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH ** 2))
    
    # 4. Density Term: Inhibition
    # 1 / (1 + (b / K_b)^n)
    # This is 1 at b=0 and decreases as b increases.
    density_term = 1.0 / (1.0 + (b / K_b) ** n_hill)
    
    # Combine Multiplicative Part
    # This captures the core biological interaction
    base_growth = global_scale * monod_s * temp_term * ph_term * density_term
    
    # Additive Polynomial Corrections
    # These allow the model to fit residual patterns not captured by the multiplicative core.
    # We use degree 1 and 2 terms for each variable, scaled by p0 and p1? 
    # Wait, we only have 10 params. 
    # p0 is currently unused in the logic above except as a placeholder. 
    # Let's reassign roles to ensure all 10 are used effectively.
    
    # Revised Parameter Roles:
    # p0: Global additive offset (intercept)
    # p1: Ks
    # p2: T_opt
    # p3: sigma_T
    # p4: pH_opt
    # p5: sigma_pH
    # p6: K_b
    # p7: n_hill
    # p8: Scale for Monod term (multiplies monod_s)
    # p9: Global multiplicative scale
    
    # Let's refine the formula to use p0 as an offset.
    
    # Recalculate with refined roles
    Ks = np.maximum(np.abs(p1), 1e-6)
    T_opt = p2
    sigma_T = np.maximum(np.abs(p3), 1e-6)
    pH_opt = p4
    sigma_pH = np.maximum(np.abs(p5), 1e-6)
    K_b = np.maximum(np.abs(p6), 1e-6)
    n_hill = np.maximum(np.abs(p7), 0.1)
    monod_scale = p8
    global_scale = p9
    
    monod_s = s / (Ks + s)
    temp_term = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T ** 2))
    ph_term = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH ** 2))
    density_term = 1.0 / (1.0 + (b / K_b) ** n_hill)
    
    # Core multiplicative model
    core = global_scale * monod_scale * monod_s * temp_term * ph_term * density_term
    
    # Additive offset
    growth = core + p0
    
    # Ensure finite
    growth = np.where(np.isfinite(growth), growth, 0.0)
    
    return growth
# EVOLVE-BLOCK-END
