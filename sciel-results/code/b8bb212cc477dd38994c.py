"""E. coli growth rate model: Monod substrate x Gaussian temp/pH with quadratic flexibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with biologically-motivated multiplicative structure.
    
    mu = p0 * (s / (p1 + s)) * exp(-0.5 * ((temp - p2) / p3)^2) * exp(-0.5 * ((pH - p4) / p5)^2)
       * (1 + p6*(temp - p2)*(pH - p4)) * (1 - p7*b) + p8*b*s + p9
    
    Parameters (10):
    p0: Max growth rate scaling
    p1: Monod half-saturation constant (Ks)
    p2: Optimal temperature
    p3: Temperature width (sigma)
    p4: Optimal pH
    p5: pH width (sigma)
    p6: Temp-pH interaction coefficient
    p7: Density inhibition coefficient
    p8: Density-substrate interaction coefficient
    p9: Bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure positive width parameters for Gaussian terms
    # Use softplus-like or just clip to avoid division by zero/negative
    # Since BFGS starts at 1 and we want smooth gradients, we can use:
    # Actually, for stability in a closed-form expression without extra ops,
    # let's rely on the optimizer not going to 0. But to be safe:
    # If p3 or p5 becomes negative, the Gaussian exponent becomes positive -> explosion.
    # We can use np.abs(p3) + 1e-6 to ensure positive width.
    sigma_t = np.abs(p3) + 1e-6
    sigma_h = np.abs(p5) + 1e-6
    
    # Monod term: s / (Ks + s)
    # If p1 (Ks) is negative and close to -s, we have issues.
    # Start at 1. Let's assume it stays positive or use abs?
    # Monod requires Ks > 0.
    Ks = np.abs(p1) + 1e-6
    monod = s / (Ks + s)
    
    # Gaussian terms for temp and pH
    # Center at p2 and p4
    t_dev = temp - p2
    h_dev = pH - p4
    
    gauss_t = np.exp(-0.5 * (t_dev / sigma_t) ** 2)
    gauss_h = np.exp(-0.5 * (h_dev / sigma_h) ** 2)
    
    # Interaction term
    # Normalized interaction to keep coefficients small
    # (t_dev * h_dev) / (sigma_t * sigma_h)
    inter_norm = (t_dev * h_dev) / (sigma_t * sigma_h)
    interaction = 1.0 + p6 * inter_norm
    
    # Density inhibition
    # b is in [1, 5]. Let's normalize b to [0, 1] for the inhibition term
    b_norm = (b - 1.0) / 4.0
    inhibition = 1.0 - p7 * b_norm
    
    # Density-substrate interaction
    # Simple product of normalized b and s
    # s is in [1, 40] roughly? Training range says [1, 5] for b, [1, 5] for s?
    # Wait, task says: inputs in training range: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # So b: [1, 5], s: [1, 5], temp: [1, 40], pH: [2, 12]
    # My previous normalization for s was wrong if s is [1,5].
    # Let's re-normalize s for the interaction term if needed, or just use raw.
    # p8 * b * s might be large. Let's normalize s too.
    s_norm = (s - 1.0) / 4.0
    density_substrate = p8 * b_norm * s_norm
    
    # Combine
    pred = p0 * monod * gauss_t * gauss_h * interaction * inhibition + density_substrate + p9
    
    # Ensure finite
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
