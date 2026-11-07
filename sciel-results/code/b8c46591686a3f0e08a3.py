"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Normalize inputs to [-1, 1] range
    # Domain is [0.5, 2.0]
    # Center = 1.25, Range = 1.5, Half-Range = 0.75
    x_c = (x - 1.25) / 0.75
    y_c = (y - 1.25) / 0.75
    z_c = (z - 1.25) / 0.75
    
    p = params
    
    # Basis functions:
    # 1. Constant
    # 2-4. Linear (P1)
    # 5-7. Quadratic (P2, orthogonal to P0 and P1)
    # 8-10. Interactions of linear terms
    
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3*t^2 - 1)
    
    p0 = 1.0
    p1_x = x_c
    p1_y = y_c
    p1_z = z_c
    
    p2_x = 0.5 * (3.0 * x_c * x_c - 1.0)
    p2_y = 0.5 * (3.0 * y_c * y_c - 1.0)
    p2_z = 0.5 * (3.0 * z_c * z_c - 1.0)
    
    pred = (p[0] * p0 +
            p[1] * p1_x +
            p[2] * p1_y +
            p[3] * p1_z +
            p[4] * p2_x +
            p[5] * p2_y +
            p[6] * p2_z +
            p[7] * (p1_x * p1_y) +
            p[8] * (p1_x * p1_z) +
            p[9] * (p1_y * p1_z))
            
    return pred
# EVOLVE-BLOCK-END
