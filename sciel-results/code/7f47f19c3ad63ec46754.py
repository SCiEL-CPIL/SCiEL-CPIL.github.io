"""Symbolic regression: E. coli growth rate with Kernel Ridge Regression (RBF) formulation."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a hybrid approach:
    - Physical priors: Monod for substrate, Gaussian for temp/pH, logistic for density
    - RBF kernel correction: weighted sum of Gaussian similarities to a fixed grid
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # --- Physical base model ---
    # Monod substrate term
    Ks = np.exp(np.clip(p1, -10, 10))
    monod = s / (Ks + s)

    # Temperature response: Gaussian
    temp_opt = p2
    temp_width = np.exp(np.clip(p3, -5, 5))
    temp_resp = np.exp(-0.5 * ((temp - temp_opt) / temp_width) ** 2)

    # pH response: Gaussian
    ph_opt = p4
    ph_width = np.exp(np.clip(p5, -5, 5))
    ph_resp = np.exp(-0.5 * ((pH - ph_opt) / ph_width) ** 2)

    # Maximum growth rate
    mu_max = np.exp(np.clip(p0, -10, 10))

    # Density effect: logistic inhibition
    K_carry = np.exp(np.clip(p6, -10, 10))
    density_factor = 1.0 - (b / (K_carry + b))

    # Base prediction
    mu_base = mu_max * monod * temp_resp * ph_resp * density_factor

    # --- RBF Kernel Correction ---
    # Fixed grid points for the kernel centers
    # Normalize inputs to [0, 1] roughly for stable kernel evaluation
    b_norm = b / (K_carry + b)  # already in [0, 1)
    s_norm = monod  # already in [0, 1)
    temp_norm = np.clip((temp - 20.0) / 20.0, -1.5, 1.5)
    ph_norm = np.clip((pH - 7.0) / 5.0, -1.5, 1.5)

    # Stack normalized inputs: (N, 4)
    X_norm = np.column_stack([b_norm, s_norm, temp_norm, ph_norm])

    # Define a small fixed grid of centers in the 4D space
    # We use a 2x2x2x2 grid for a total of 16 centers, but we only have 4 params (p7-p10)
    # So let's use 4 centers, or use the 4 params as weights for 4 specific centers.
    # Actually, let's use a more flexible approach:
    # Use 4 RBF centers defined by fixed coordinates, weighted by p7, p8, p9, and (p0? no, p0 is mu_max)
    # Let's use p7, p8, p9 and a new combination for the 4th weight? Or just 3 centers?
    # Let's use 4 centers:
    # Center 1: low b, low s, optimal temp, optimal pH
    # Center 2: high b, high s, low temp, low pH
    # Center 3: low b, high s, high temp, high pH
    # Center 4: high b, low s, low temp, high pH
    
    # Define centers in normalized space
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],   # c1
        [1.0, 1.0, -1.0, -1.0], # c2
        [0.0, 1.0, 1.0, 1.0],   # c3
        [1.0, 0.0, -1.0, 1.0]   # c4
    ])
    
    # Kernel width
    gamma = 0.5 / 4.0  # Standard RBF kernel width
    
    # Compute kernel values: exp(-gamma * ||x - center||^2)
    # X_norm is (N, 4), centers is (4, 4)
    # Distances: (N, 4, 4)
    diffs = X_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)  # (N, 4)
    
    # Kernel matrix
    K = np.exp(-gamma * dist_sq)  # (N, 4)
    
    # Weights for the 4 centers: p7, p8, p9, and use p0? No, p0 is critical for mu_max.
    # Let's use p7, p8, p9 and a linear combination of p1, p2? No, keep it simple.
    # We have 10 params. p0-p6 are used in base. p7, p8, p9 are free.
    # We need 4 weights. Let's use p7, p8, p9 and (p7 + p8 + p9)? Or just 3 centers?
    # Let's use 3 centers with weights p7, p8, p9.
    # Or better: use 4 centers, with weights p7, p8, p9, and (p7*p8? no, keep linear)
    # Let's use p7, p8, p9 and p10? No, only 10 params (p0-p9).
    # So we have 3 free params p7, p8, p9.
    # Let's use 3 RBF centers.
    
    centers_3 = centers[:3]
    diffs_3 = X_norm[:, np.newaxis, :] - centers_3[np.newaxis, :, :]
    dist_sq_3 = np.sum(diffs_3 ** 2, axis=2)  # (N, 3)
    K_3 = np.exp(-gamma * dist_sq_3)  # (N, 3)
    
    weights = np.array([p7, p8, p9])
    rbf_correction = K_3 @ weights  # (N,)
    
    # Combine
    result = mu_base + rbf_correction
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
