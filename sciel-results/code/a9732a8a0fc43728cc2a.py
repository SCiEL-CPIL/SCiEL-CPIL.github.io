"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1] for Legendre polynomials
    # Inputs are in [0.5, 2.0], center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Define Shifted Legendre Polynomials P0(t), P1(t), P2(t), P3(t)
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    # P3(t) = 0.5 * (5t^3 - 3t)
    
    # Compute for each input
    Pu0 = 1.0
    Pu1 = u
    Pu2 = 0.5 * (3.0 * u * u - 1.0)
    Pu3 = 0.5 * (5.0 * u * u * u - 3.0 * u)
    
    Pv0 = 1.0
    Pv1 = v
    Pv2 = 0.5 * (3.0 * v * v - 1.0)
    Pv3 = 0.5 * (5.0 * v * v * v - 3.0 * v)
    
    Pw0 = 1.0
    Pw1 = w
    Pw2 = 0.5 * (3.0 * w * w - 1.0)
    Pw3 = 0.5 * (5.0 * w * w * w - 3.0 * w)
    
    # We have 10 parameters. 
    # Strategy: Use the most significant orthogonal terms.
    # Indices:
    # 0: Constant (P0*P0*P0)
    # 1: Linear x (Pu1)
    # 2: Linear y (Pv1)
    # 3: Linear z (Pw1)
    # 4: Quadratic x (Pu2)
    # 5: Quadratic y (Pv2)
    # 6: Quadratic z (Pw2)
    # 7: Interaction xy (Pu1 * Pv1)
    # 8: Interaction xz (Pu1 * Pw1)
    # 9: Interaction yz (Pv1 * Pw1)
    
    # Note: This leaves out cubic terms (Pu3, etc.) and higher-order interactions,
    # but ensures the basis is orthogonal where possible and well-conditioned.
    # If the function has significant cubic behavior, this might be a limitation,
    # but the conditioning gain from Legendre vs raw polynomials is substantial for BFGS.
    
    return (p[0] * Pu0 * Pv0 * Pw0 +
            p[1] * Pu1 * Pv0 * Pw0 +
            p[2] * Pu0 * Pv1 * Pw0 +
            p[3] * Pu0 * Pv0 * Pw1 +
            p[4] * Pu2 * Pv0 * Pw0 +
            p[5] * Pu0 * Pv2 * Pw0 +
            p[6] * Pu0 * Pv0 * Pw2 +
            p[7] * Pu1 * Pv1 * Pw0 +
            p[8] * Pu1 * Pv0 * Pw1 +
            p[9] * Pu0 * Pv1 * Pw1)
# EVOLVE-BLOCK-END
