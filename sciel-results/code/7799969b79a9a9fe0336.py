"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network for nonlinear oscillator.
    10 basis functions: 5 in (x,v) space, 5 in (t) space, plus a constant.
    Centers and widths are fixed; only 10 weights are fitted.
    Training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624], t in [30, 50]
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # RBF centers in (x, v) space - spread across the domain
    # x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    # Widths scaled to domain size for smooth coverage
    
    # 5 RBFs in (x, v) space
    # Center 1: low x, low v
    cx1, cv1 = -0.15, -0.25
    # Center 2: mid x, mid v
    cx2, cv2 = 0.0, 0.0
    # Center 3: high x, low v
    cx3, cv3 = 0.15, -0.25
    # Center 4: low x, high v
    cx4, cv4 = -0.15, 0.25
    # Center 5: high x, high v
    cx5, cv5 = 0.15, 0.25
    
    # Widths: x width ~0.15, v width ~0.25 (scaled to domain)
    sx = 0.15
    sv = 0.25
    
    dx1 = (x - cx1) / sx
    dv1 = (v - cv1) / sv
    dx2 = (x - cx2) / sx
    dv2 = (v - cv2) / sv
    dx3 = (x - cx3) / sx
    dv3 = (v - cv3) / sv
    dx4 = (x - cx4) / sx
    dv4 = (v - cv4) / sv
    dx5 = (x - cx5) / sx
    dv5 = (v - cv5) / sv
    
    rbf1 = np.exp(-0.5 * (dx1**2 + dv1**2))
    rbf2 = np.exp(-0.5 * (dx2**2 + dv2**2))
    rbf3 = np.exp(-0.5 * (dx3**2 + dv3**2))
    rbf4 = np.exp(-0.5 * (dx4**2 + dv4**2))
    rbf5 = np.exp(-0.5 * (dx5**2 + dv5**2))
    
    # 4 RBFs in t space (t in [30, 50], width ~5)
    st = 5.0
    dt1 = (t - 32.0) / st
    dt2 = (t - 38.0) / st
    dt3 = (t - 44.0) / st
    dt4 = (t - 48.0) / st
    
    trbf1 = np.exp(-0.5 * dt1**2)
    trbf2 = np.exp(-0.5 * dt2**2)
    trbf3 = np.exp(-0.5 * dt3**2)
    trbf4 = np.exp(-0.5 * dt4**2)
    
    # p0 is a constant bias term
    result = (
        p0
        + p1 * rbf1
        + p2 * rbf2
        + p3 * rbf3
        + p4 * rbf4
        + p5 * rbf5
        + p6 * trbf1
        + p7 * trbf2
        + p8 * trbf3
        + p9 * trbf4
    )
    
    return result
# EVOLVE-BLOCK-END
