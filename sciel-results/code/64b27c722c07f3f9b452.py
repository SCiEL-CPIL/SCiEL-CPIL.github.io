"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a biologically motivated mechanistic model.
    
    Strategy:
    Instead of a generic linear or black-box model, we use a product of biologically
    meaningful factors:
    1. Substrate limitation: Monod equation (s / (Ks + s))
    2. Temperature effect: Gaussian around an optimum (exp(-((T - Topt)^2) / (2 * sigma_T^2)))
    3. pH effect: Gaussian around an optimum (exp(-((pH - pHo)^2) / (2 * sigma_pH^2)))
    4. Population density: Linear term (b) representing potential inhibition or growth
    
    The model is multiplicative to capture interactions (e.g., substrate effect depends on T).
    
    Parameters (10 total):
    p[0]: Max growth rate scaling (mu_max)
    p[1]: Monod constant (Ks)
    p[2]: Temperature optimum (Topt)
    p[3]: Temperature standard deviation (sigma_T)
    p[4]: pH optimum (pHo)
    p[5]: pH standard deviation (sigma_pH)
    p[6]: Population density coefficient (linear)
    p[7]: Baseline offset
    p[8]: Additional substrate interaction term
    p[9]: Additional temperature interaction term
    """
    mu_max = params[0]
    Ks = np.abs(params[1]) + 1e-6  # Ensure positive Ks
    Topt = params[2]
    sigma_T = np.abs(params[3]) + 1e-6  # Ensure positive sigma
    pHo = params[4]
    sigma_pH = np.abs(params[5]) + 1e-6  # Ensure positive sigma
    c_b = params[6]
    offset = params[7]
    c_s_int = params[8]
    c_t_int = params[9]
    
    # Monod term for substrate
    monod_s = s / (Ks + s)
    
    # Gaussian temperature term
    temp_term = np.exp(-((temp - Topt) ** 2) / (2 * sigma_T ** 2))
    
    # Gaussian pH term
    ph_term = np.exp(-((pH - pHo) ** 2) / (2 * sigma_pH ** 2))
    
    # Base growth rate from environmental factors
    base_growth = mu_max * monod_s * temp_term * ph_term
    
    # Add linear population density effect and interactions
    # We use a weighted sum of the base growth and linear terms to allow flexibility
    # while keeping the mechanistic core.
    
    # To utilize all 10 params meaningfully and allow for non-multiplicative effects:
    # We can add small linear corrections or scale the density term differently.
    
    # Let's define the final prediction as a combination:
    # pred = base_growth * (1 + c_b * b) + offset + c_s_int * s * temp_term + c_t_int * temp * ph_term
    
    # This allows the model to capture:
    # 1. Core mechanistic saturation and optima (base_growth)
    # 2. Density-dependent inhibition/promotion (1 + c_b * b)
    # 3. Direct linear effects of s and temp modulated by other factors
    # 4. A global offset
    
    density_factor = 1.0 + c_b * b
    
    # Interaction terms:
    # Substrate effect might be slightly different at different temps, but Monod handles s.
    # Let's add a term that scales with s and temp to capture potential synergies not in Monod.
    term_s_t = c_s_int * s * temp_term
    term_t_p = c_t_int * temp * ph_term
    
    prediction = base_growth * density_factor + offset + term_s_t + term_t_p
    
    return prediction
# EVOLVE-BLOCK-END
