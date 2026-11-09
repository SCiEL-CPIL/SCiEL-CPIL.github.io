"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 3x3 grid of Gaussian RBFs + 1 constant term
    # Centers at -1, 0, 1 for both x and y
    # Fixed width sigma^2 = 0.5 (exponent factor 1.0)
    
    # Constant term
    c0 = params[0]
    
    # Gaussian terms
    # We will unroll the 3x3 grid for speed and clarity
    # Centers:
    # (x_c, y_c) in {(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)}
    
    # Row 1: y_c = -1
    g0 = np.exp(-((x - (-1.0))**2 + (y - (-1.0))**2))
    g1 = np.exp(-((x - (-1.0))**2 + (y - 0.0)**2))
    g2 = np.exp(-((x - (-1.0))**2 + (y - 1.0)**2))
    
    # Row 2: y_c = 0
    g3 = np.exp(-((x - 0.0)**2 + (y - (-1.0))**2))
    g4 = np.exp(-((x - 0.0)**2 + (y - 0.0)**2))
    g5 = np.exp(-((x - 0.0)**2 + (y - 1.0)**2))
    
    # Row 3: y_c = 1
    g6 = np.exp(-((x - 1.0)**2 + (y - (-1.0))**2))
    g7 = np.exp(-((x - 1.0)**2 + (y - 0.0)**2))
    g8 = np.exp(-((x - 1.0)**2 + (y - 1.0)**2))
    
    # Weights
    w0 = params[1]
    w1 = params[2]
    w2 = params[3]
    w3 = params[4]
    w4 = params[5]
    w5 = params[6]
    w6 = params[7]
    w7 = params[8]
    w8 = params[9]
    
    return (c0 + 
            w0 * g0 + w1 * g1 + w2 * g2 + 
            w3 * g3 + w4 * g4 + w5 * g5 + 
            w6 * g6 + w7 * g7 + w8 * g8)
# EVOLVE-BLOCK-END
