"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Extract parameters with safe defaults
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe denominators to avoid division by zero or negative values in log/exp
    Ks = np.maximum(p1, 1e-6)
    sigma_T = np.maximum(np.abs(p3), 1e-6)
    sigma_pH = np.maximum(np.abs(p5), 1e-6)
    K_b = np.maximum(np.abs(p6), 1e-6)
    
    # Monod kinetics for substrate: s / (Ks + s)
    # This is bounded [0, 1] for s >= 0, providing a natural saturation.
    monod_s = s / (Ks + s)
    
    # Temperature response: Gaussian around optimal temperature T_opt
    # Using a Gaussian is robust and smooth for BFGS.
    # exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
    temp_resp = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH response: Gaussian around optimal pH
    # E. coli typically has an optimal pH around 7.
    ph_resp = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density response: Logistic inhibition
    # As density b increases, growth rate decreases.
    # 1 - b/K is standard, but can go negative. 
    # Using exp(-b/K) is always positive and smooth, which is better for optimizers.
    # Alternatively, a soft logistic: 1 / (1 + exp(p7 * (b - K_b)))
    # Let's use a simple exponential decay for stability: exp(-p7 * b / K_b)
    # Or simply (1 - b/K_b) clipped to be non-negative? 
    # The inspiration used max(0, 1-b/K) and exp(-inhibit*b).
    # Let's use a smooth logistic-like term that stays positive:
    # 1 / (1 + exp(p7 * (b / K_b)))
    # This transitions from 1 to 0 as b increases.
    density_resp = 1.0 / (1.0 + np.exp(p7 * (b / K_b)))
    
    # Base growth rate
    growth = p0 * monod_s * temp_resp * ph_resp * density_resp
    
    # Add interaction terms or offsets if needed, but keep it simple first.
    # The linear model had an offset. Let's add p8 as a baseline offset.
    growth += p8
    
    # p9 could be used for a secondary term, e.g., interaction between temp and pH?
    # Or just left as a small coefficient for a linear term in s?
    # To maximize flexibility without breaking smoothness, let's add a small linear term for s if p9 is non-zero.
    # Actually, let's keep it purely multiplicative for the main terms and use p9 for a fine-tuning linear term in temp?
    # No, let's stick to the mechanistic form. If p9 is not used, it will be optimized to 0 or stay at 1.
    # Better: use p9 to scale the density inhibition strength? No, p7 does that.
    # Let's use p9 as a scaling factor for the Monod term?
    # growth *= (1 + p9 * monod_s) ?
    # Let's just add a small linear term in s to allow for non-saturating behavior if needed?
    # growth += p9 * s ?
    # This might destabilize if s is large.
    # Let's keep p9 as a multiplier for the entire growth rate?
    # growth *= p9 ?
    # That's redundant with p0.
    # Let's use p9 to adjust the temperature sensitivity?
    # No, let's just ensure all 10 params are used in a stable way.
    # p0: max rate
    # p1: Ks
    # p2: T_opt
    # p3: sigma_T
    # p4: pH_opt
    # p5: sigma_pH
    # p6: K_b (density carrying capacity)
    # p7: density inhibition sharpness
    # p8: offset
    # p9: Let's use it to scale the substrate term slightly, allowing for some linear component.
    # monod_s_eff = monod_s + p9 * (s / (1 + s)) ?
    # Let's just multiply the final result by p9 to allow global scaling? 
    # Wait, p0 already scales it.
    # Let's use p9 to modify the pH tolerance?
    # Let's just add p9 * temp * pH as a small interaction?
    # growth += p9 * temp * pH
    # This is safe and linear.
    
    growth += p9 * temp * pH
    
    # Ensure finite values
    growth = np.where(np.isfinite(growth), growth, 0.0)
    
    return growth
# EVOLVE-BLOCK-END
