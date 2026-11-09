"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: [E, yield_stress, hardening_rate, saturation_strain, temp_coeff_yield,
    #          temp_coeff_hard, elastic_offset, plastic_offset, blend_width, global_scale]
    E = params[0]           # Elastic modulus
    sy = params[1]          # Yield stress at reference temp
    H = params[2]           # Hardening rate
    eps_sat = params[3]     # Strain scale for saturation
    c_ty = params[4]        # Temp coefficient for yield
    c_th = params[5]        # Temp coefficient for hardening
    e_off = params[6]       # Elastic offset
    p_off = params[7]       # Plastic offset
    w = params[8]           # Blend width (transition sharpness)
    g_scale = params[9]     # Global scale

    # Temperature-adjusted yield and hardening
    sy_t = sy * (1.0 + c_ty * temp)
    H_t = H * (1.0 + c_th * temp)

    # Elastic stress
    stress_elastic = E * strain + e_off

    # Plastic stress: yield + hardening * (strain - yield_strain) but saturating
    # Use a smooth saturation: H_t * eps_sat * tanh((strain - sy_t/E) / eps_sat)
    strain_yield = sy_t / np.maximum(E, 1e-6)
    plastic_term = H_t * eps_sat * np.tanh((strain - strain_yield) / np.maximum(eps_sat, 1e-6))
    stress_plastic = sy_t + plastic_term + p_off

    # Smooth blend between elastic and plastic based on strain relative to yield
    # Use a sigmoid-like blend: t = 1/(1+exp(-k*(strain-strain_yield)))
    k = 1.0 / np.maximum(w, 1e-6)
    t = 1.0 / (1.0 + np.exp(-k * (strain - strain_yield)))
    
    # Clip t to avoid numerical issues
    t = np.clip(t, 0.0, 1.0)

    # Blend
    stress = (1.0 - t) * stress_elastic + t * stress_plastic
    stress = g_scale * stress

    return stress
# EVOLVE-BLOCK-END
