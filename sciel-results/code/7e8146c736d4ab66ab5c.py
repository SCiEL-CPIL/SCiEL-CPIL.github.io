# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF Network with 3x3 grid of Gaussians
    # params[0] is log(inverse_variance)
    # params[1:10] are weights, shifted by -1 to start at 0
    
    # alpha = exp(params[0])
    # Starts at exp(1) = 2.718
    alpha = np.exp(params[0])
    
    # Weights start at 0 because params start at 1
    w0 = params[1] - 1.0
    w1 = params[2] - 1.0
    w2 = params[3] - 1.0
    w3 = params[4] - 1.0
    w4 = params[5] - 1.0
    w5 = params[6] - 1.0
    w6 = params[7] - 1.0
    w7 = params[8] - 1.0
    w8 = params[9] - 1.0
    
    x2 = x * x
    y2 = y * y
    
    r2_00 = x2 + y2 + 2.0*x + 2.0*y + 2.0
    g0 = np.exp(-alpha * r2_00)
    
    r2_01 = x2 + y2 + 2.0*x + 1.0
    g1 = np.exp(-alpha * r2_01)
    
    r2_02 = x2 + y2 + 2.0*x - 2.0*y + 2.0
    g2 = np.exp(-alpha * r2_02)
    
    r2_10 = x2 + y2 + 2.0*y + 1.0
    g3 = np.exp(-alpha * r2_10)
    
    r2_11 = x2 + y2
    g4 = np.exp(-alpha * r2_11)
    
    r2_12 = x2 + y2 - 2.0*y + 1.0
    g5 = np.exp(-alpha * r2_12)
    
    r2_20 = x2 + y2 - 2.0*x + 2.0*y + 2.0
    g6 = np.exp(-alpha * r2_20)
    
    r2_21 = x2 + y2 - 2.0*x + 1.0
    g7 = np.exp(-alpha * r2_21)
    
    r2_22 = x2 + y2 - 2.0*x - 2.0*y + 2.0
    g8 = np.exp(-alpha * r2_22)
    
    return (w0 * g0 + w1 * g1 + w2 * g2 + 
            w3 * g3 + w4 * g4 + w5 * g5 + 
            w6 * g6 + w7 * g7 + w8 * g8)
    # EVOLVE-BLOCK-END
