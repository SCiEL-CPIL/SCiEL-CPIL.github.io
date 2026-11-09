"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: combines normalized linear terms, log-space terms, and
    log-space interaction terms to capture additive, power-law, and
    multiplicative structures with 10 parameters.
    
    Parameters (10 total):
    0: constant
    1: u (normalized x)
    2: v (normalized y)
    3: w (normalized z)
    4: lx (log x)
    5: ly (log y)
    6: lz (log z)
    7: lx * ly (log interaction x,y)
    8: lx * lz (log interaction x,z)
    9: ly * lz (log interaction y,z)
    """
    p = params
    
    # Normalize inputs to [-1, 1] for better conditioning
    # x, y, z are in [0.5, 2.0]
    # center = 1.25, scale = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Log-transformed inputs (safe since inputs in [0.5, 2])
    x_safe = np.clip(x, 1e-8, None)
    y_safe = np.clip(y, 1e-8, None)
    z_safe = np.clip(z, 1e-8, None)
    
    lx = np.log(x_safe)
    ly = np.log(y_safe)
    lz = np.log(z_safe)
    
    # Basis terms:
    # 0: constant
    # 1: u (linear x normalized)
    # 2: v (linear y normalized)
    # 3: w (linear z normalized)
    # 4: lx (log x)
    # 5: ly (log y)
    # 6: lz (log z)
    # 7: lx * ly (interaction in log-space)
    # 8: lx * lz (interaction in log-space)
    # 9: ly * lz (interaction in log-space)
    
    return (p[0] +
            p[1] * u +
            p[2] * v +
            p[3] * w +
            p[4] * lx +
            p[5] * ly +
            p[6] * lz +
            p[7] * lx * ly +
            p[8] * lx * lz +
            p[9] * ly * lz)
# EVOLVE-BLOCK-END
