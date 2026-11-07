import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative Gaussian RBF Model.
    Assumes growth rate is a product of optimal response functions for each input.
    p0: Baseline offset
    p1: Amplitude scale
    p2, p3: Width and Center for b (normalized)
    p4, p5: Width and Center for s (normalized)
    p6, p7: Width and Center for temp (normalized)
    p8, p9: Width and Center for pH (normalized)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to ~[-1, 1] range to stabilize optimization
    # b in [1, 5] -> center 3, scale 2
    b_norm = (b - 3.0) / 2.0
    # s in [1, 5] -> center 3, scale 2
    s_norm = (s - 3.0) / 2.0
    # temp in [1, 40] -> center 20.5, scale 19.5
    t_norm = (temp - 20.5) / 19.5
    # pH in [2, 12] -> center 7, scale 5
    ph_norm = (pH - 7.0) / 5.0
    
    # Ensure widths are positive to avoid numerical issues with exp
    # Using exp(params) ensures positivity, but params start at 1, so exp(1)=2.7
    # If we just use params directly, they can go negative, making exp(+inf) if width is negative?
    # exp(-width * x^2). If width is negative, exp(+x^2) -> overflow.
    # So we must ensure width > 0.
    # Let's use exp(width_param) for the width coefficient in the exponent.
    # So term is exp(-exp(p2) * (b_norm - p3)^2)
    
    width_b = np.exp(p2)
    width_s = np.exp(p4)
    width_t = np.exp(p6)
    width_ph = np.exp(p8)
    
    gauss_b = np.exp(-width_b * (b_norm - p3)**2)
    gauss_s = np.exp(-width_s * (s_norm - p5)**2)
    gauss_t = np.exp(-width_t * (t_norm - p7)**2)
    gauss_ph = np.exp(-width_ph * (ph_norm - p9)**2)
    
    pred = p0 + p1 * gauss_b * gauss_s * gauss_t * gauss_ph
    
    return pred
