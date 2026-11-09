"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid model: polynomial basis (degree 5) + RBF bump functions
    # params[0:6]  -> coefficients for x^0 .. x^5
    # params[6:10] -> widths for 4 RBF centers at fixed positions
    # RBF weights are derived from polynomial tail to keep param count at 10

    x = np.asarray(x, dtype=np.float64)

    # Polynomial part: degree 5
    p_coeffs = params[0:6]
    # Horner-like evaluation for stability
    poly = np.zeros_like(x)
    for i in range(5, -1, -1):
        poly = poly * x + p_coeffs[i]

    # RBF part: 4 fixed centers, learnable widths via softplus
    centers = np.array([-2.0, -0.7, 0.7, 2.0], dtype=np.float64)
    widths_raw = params[6:10]
    # softplus for positive widths: log(1 + exp(t)), numerically stable
    widths = np.logaddexp(0.0, widths_raw)  # softplus
    # Clamp to reasonable range to avoid overflow/underflow
    widths = np.clip(widths, 0.05, 5.0)

    # RBF weights: use a fixed pattern scaled by a global gain param? 
    # We only have 4 width params left; let's use fixed weights but modulated by width
    # Actually, let's use the polynomial coefficients to also inform RBF weights
    # Better: use 2 RBF weights from remaining params? We have exactly 10 params.
    # Let's restructure: 4 poly coeffs (x^0..x^3) + 3 RBF widths + 3 RBF weights
    # That's 4+3+3=10. But poly degree 3 may be too low.
    
    # Revised: 5 poly coeffs (x^0..x^4) + 5 RBF widths
    # RBF weights derived as: w_i = exp(-widths_i^2) * sign pattern
    # This is getting complicated. Let's use a cleaner split:
    # params[0:5] = poly coeffs (degree 4)
    # params[5:10] = 5 RBF widths at 5 fixed centers
    # RBF weights = 1.0 (fixed), so the bump height is controlled by width only
    
    # Actually, simpler and more effective:
    # params[0:6] = poly coeffs (x^0..x^5)  
    # params[6:10] = 4 widths for RBFs at [-1.5, -0.5, 0.5, 1.5]
    # RBF weight for each = 1.0, but scaled by exp(-width) to create localized bumps

    # Let me just do: 6 poly + 4 widths + 2 more params as RBF weights
    # But we need to use exactly 10.
    
    # Final clean design:
    # params[0:6]: polynomial coefficients c0..c5 for x^0..x^5
    # params[6:8]: widths for RBF centers at -1.0 and 1.0  
    # params[8:10]: weights for those two RBFs
    # Total: 6 + 2 + 2 = 10

    poly_coeffs = params[0:6]
    # Re-evaluate polynomial
    poly = np.zeros_like(x)
    for i in range(5, -1, -1):
        poly = poly * x + poly_coeffs[i]

    rbf_centers = np.array([-1.0, 1.0], dtype=np.float64)
    rbf_widths_raw = params[6:8]
    rbf_weights = params[8:10]
    
    # softplus for widths
    rbf_widths = np.logaddexp(0.0, rbf_widths_raw)
    rbf_widths = np.clip(rbf_widths, 0.1, 3.0)

    rbf_val = np.zeros_like(x)
    for c, w, wt in zip(rbf_centers, rbf_widths, rbf_weights):
        rbf_val += wt * np.exp(-0.5 * ((x - c) / w) ** 2)

    return poly + rbf_val
# EVOLVE-BLOCK-END
