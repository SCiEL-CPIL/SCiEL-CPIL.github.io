"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a bilinear elastic-plastic model with smooth transition and
    power-law hardening, all with temperature dependence.
    
    Model:
        E(T)  = p0 + p1*T       (elastic modulus)
        s_y(T) = p2 + p3*T      (yield strain)
        Y(T)  = p4 + p5*T       (yield stress level)
        n(T)  = exp(p6 + p7*T)  (hardening exponent, > 0)
        H(T)  = p8 + p9*T       (hardening modulus scale)
        
        For strain > 0:
            elastic_stress = E(T) * strain
            plastic_stress = Y(T) + H(T) * (strain - s_y(T))**n(T)  for strain > s_y(T)
            stress = smooth_min(elastic_stress, plastic_stress) via tanh blending
        
        For strain < 0:
            stress = E(T) * strain  (linear elastic)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    E = p0 + p1 * temp                    # Elastic modulus
    s_y = p2 + p3 * temp                  # Yield strain (positive)
    Y = p4 + p5 * temp                    # Yield stress
    n_raw = p6 + p7 * temp
    n_raw = np.clip(n_raw, -10.0, 10.0)   # Clamp to prevent overflow
    n = np.exp(n_raw)                     # Hardening exponent (always > 0)
    H = p8 + p9 * temp                    # Hardening scale
    
    # Ensure yield strain is positive
    s_y = np.maximum(s_y, 1e-6)
    
    # Elastic stress (valid for all strain)
    elastic_stress = E * strain
    
    # For positive strain: compute plastic stress
    # plastic_stress = Y + H * (strain - s_y)^n for strain > s_y
    # For strain <= s_y, plastic_stress should be >= elastic to make min work
    
    # Use np.where to handle the piecewise safely
    # delta = strain - s_y (only meaningful for strain > 0)
    delta = strain - s_y
    
    # For negative delta, (delta)^n is problematic if n is non-integer
    # Use sign-preserving power: sign(delta)*|delta|^n, but we only want it for delta > 0
    # For delta <= 0, set plastic contribution to 0 so plastic_stress = Y
    # Actually, we want: if strain > s_y, plastic kicks in; else plastic is "inactive"
    
    # Safe computation of (strain - s_y)^n only where strain > s_y and strain > 0
    pos_mask = (strain > s_y) & (strain > 0)
    
    # Initialize plastic stress as Y (plateau) everywhere
    plastic_stress = np.full_like(strain, Y)
    
    # Where strain > s_y, add hardening
    if np.any(pos_mask):
        delta_pos = strain[pos_mask] - s_y[pos_mask]
        # delta_pos > 0, so power is safe
        hardening = H[pos_mask] * np.power(delta_pos, n[pos_mask])
        plastic_stress[pos_mask] = Y[pos_mask] + hardening
    
    # Now blend: for strain < 0, use elastic only
    # For strain >= 0, use smooth minimum of elastic and plastic
    
    # Use a smooth min via: min(a,b) = (a+b - |a-b|)/2, but smooth with tanh
    # Or simpler: use a soft minimum: -ln(exp(-a) + exp(-b)) but that's for log-sum-exp
    
    # Alternative: use a blending parameter based on how far past yield
    # For strain <= 0: stress = elastic
    # For strain > 0: blend between elastic and plastic
    
    # Simple approach: for strain > 0, stress = min(elastic, plastic) smoothed
    # Use: stress = (elastic + plastic - |elastic - plastic|) / 2  (exact min)
    # But we want smooth. Use: 0.5*(a+b) - 0.5*sqrt((a-b)^2 + eps)  (smooth min)
    
    eps = 1e-8
    diff = elastic_stress - plastic_stress
    smooth_min = 0.5 * (elastic_stress + plastic_stress) - 0.5 * np.sqrt(diff * diff + eps)
    
    # For negative strain, just use elastic
    stress = np.where(strain > 0, smooth_min, elastic_stress)
    
    return stress
# EVOLVE-BLOCK-END
