"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a piecewise-linear 'growth plateau' structure.
    
    This model uses hard constraints (clipping) to enforce zero growth outside
    biologically plausible temperature and pH ranges, combined with a rational
    substrate saturation term and density-dependent inhibition.
    
    Structure:
    - Temperature gate: linear ramp up to optimum, then linear ramp down (piecewise linear)
    - pH gate: similar piecewise linear structure
    - Substrate: Monod-like saturation
    - Density: rational inhibition term
    
    The model is linear in parameters, allowing BFGS to converge reliably.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # --- Fixed biological parameters ---
    T_opt = 37.0      # Optimal temperature
    T_min = 15.0      # Minimum viable temperature
    T_max = 45.0      # Maximum viable temperature
    
    pH_opt = 7.0      # Optimal pH
    pH_min = 5.0      # Minimum viable pH
    pH_max = 9.0      # Maximum viable pH
    
    Ks = 1.0          # Substrate half-saturation constant

    # --- Piecewise-linear temperature gate ---
    # Rises linearly from T_min to T_opt, falls linearly from T_opt to T_max
    # Below T_min or above T_max: zero
    temp_gate = np.zeros_like(temp, dtype=np.float64)
    
    # Rising phase: T_min to T_opt
    mask_rise = (temp >= T_min) & (temp <= T_opt)
    temp_gate[mask_rise] = (temp[mask_rise] - T_min) / (T_opt - T_min)
    
    # Falling phase: T_opt to T_max
    mask_fall = (temp > T_opt) & (temp <= T_max)
    temp_gate[mask_fall] = (T_max - temp[mask_fall]) / (T_max - T_opt)
    
    # Outside [T_min, T_max]: remains 0

    # --- Piecewise-linear pH gate ---
    ph_gate = np.zeros_like(pH, dtype=np.float64)
    
    mask_ph_rise = (pH >= pH_min) & (pH <= pH_opt)
    ph_gate[mask_ph_rise] = (pH[mask_ph_rise] - pH_min) / (pH_opt - pH_min)
    
    mask_ph_fall = (pH > pH_opt) & (pH <= pH_max)
    ph_gate[mask_ph_fall] = (pH_max - pH[mask_ph_fall]) / (pH_max - pH_opt)

    # --- Substrate saturation (Monod) ---
    monod_s = s / (Ks + s)

    # --- Density inhibition (rational form) ---
    # Normalize density for conditioning: b in [1, 5]
    b_n = (b - 3.0) / 2.0
    density_inhib = 1.0 / (1.0 + b_n * b_n + 1.0)

    # --- Construct additive basis (linear in params) ---
    # p0: Bias
    pred = np.full_like(b, p0, dtype=np.float64)
    
    # p1: Substrate effect
    pred += p1 * monod_s
    
    # p2: Temperature gate
    pred += p2 * temp_gate
    
    # p3: pH gate
    pred += p3 * ph_gate
    
    # p4: Substrate * Temperature interaction
    pred += p4 * monod_s * temp_gate
    
    # p5: Substrate * pH interaction
    pred += p5 * monod_s * ph_gate
    
    # p6: Temperature * pH interaction
    pred += p6 * temp_gate * ph_gate
    
    # p7: Density inhibition
    pred += p7 * density_inhib
    
    # p8: Substrate * Density interaction
    pred += p8 * monod_s * density_inhib
    
    # p9: Triple interaction: Substrate * Temperature * pH
    pred += p9 * monod_s * temp_gate * ph_gate

    return pred
# EVOLVE-BLOCK-END
