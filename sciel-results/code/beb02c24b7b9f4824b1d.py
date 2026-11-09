"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Structural design: Multiplicative physics-informed structure with normalized inputs.
    This approach gives BFGS a well-conditioned landscape where each parameter has a
    distinct physical role, avoiding the ill-conditioned correlated-coefficient problem
    of additive linear bases.
    
    Model: a = (k0 + k1*t_n) * x + (c0 + c1*v_n) * v + F0*sin(t + phi) + F1*cos(t + phi)
    
    Parameters:
    p0: base stiffness (k0)
    p1: stiffness modulation by time (k1)
    p2: base damping (c0)
    p3: damping modulation by velocity (c1)
    p4: sine drive amplitude (F0)
    p5: cosine drive amplitude (F1)
    p6: drive phase (phi)
    p7: cubic stiffness (Duffing term)
    p8: bias
    p9: reserved (set to 0 effectively by not using it in a way that breaks symmetry)
    """
    # Normalize inputs to roughly [-1, 1] scale to help BFGS convergence
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Extract parameters
    k0 = params[0]
    k1 = params[1]
    c0 = params[2]
    c1 = params[3]
    F0 = params[4]
    F1 = params[5]
    phi = params[6]
    k3 = params[7]
    bias = params[8]
    # params[9] is unused in this formulation; it will be fitted to minimize MSE
    # but since it doesn't appear in the output, BFGS will try to drive it to 0
    # or it will have no gradient. To be safe, we can add a tiny term to keep it active
    # or just ignore it. Let's add a tiny linear term in t_n to give it a role.
    # Actually, better to use it for a higher-order coupling to keep the model rich.
    # Let's use p9 for a quadratic damping term: p9 * v_n * abs(v_n)
    nonlinear_damping = params[9] * v_n * np.abs(v_n)
    
    # Compute features
    # Stiffness force: (k0 + k1*t_n) * x_n + k3 * x_n^3
    stiffness_force = (k0 + k1 * t_n) * x_n + k3 * x_n * x_n * x_n
    
    # Damping force: (c0 + c1*v_n) * v_n + nonlinear_damping
    damping_force = (c0 + c1 * v_n) * v_n + nonlinear_damping
    
    # Driving force: F0*sin(t + phi) + F1*cos(t + phi)
    sin_drive = np.sin(t + phi)
    cos_drive = np.cos(t + phi)
    driving_force = F0 * sin_drive + F1 * cos_drive
    
    # Total acceleration
    return bias + stiffness_force + damping_force + driving_force
# EVOLVE-BLOCK-END
