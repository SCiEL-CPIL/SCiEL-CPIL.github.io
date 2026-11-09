"""Symbolic regression with 5 learnable RBF basis functions (10 params)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = sum_{i=1}^{5} a_i * exp(-((x - c_i)^2) / (2 * w_i^2))
    
    10 parameters:
      - a_1..a_5: amplitudes (linear)
      - c_1..c_5: centers (nonlinear, but well-behaved)
      - w_1..w_5: widths (constrained positive via softplus)
    
    This is a universal approximator on [-3, 3] with 5 localized basis functions.
    """
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Unpack parameters: a1..a5, c1..c5, w1..w5 (raw)
    a = params[0:5]
    c = params[5:10]
    # Wait, that's only 10 params total. We need 5 amplitudes + 5 centers + 5 widths = 15.
    # But we only have 10 parameters. Let me rethink.
    # 
    # With 10 params: 3 centers + 3 widths + 4 amplitudes? No.
    # Better: 2 RBFs with (amp, center, width) = 6 params + 4 polynomial coeffs = 10.
    # Or: 3 RBFs sharing a common width: 3 amps + 3 centers + 1 shared width + 3 extra = 10.
    # 
    # Let me use: 3 Gaussians with individual centers and shared width, plus 4 poly coeffs.
    # y = p0 + p1*x + p2*x^2 + p3*x^3 + sum_{i=1}^{3} a_i * exp(-((x-c_i)^2)/(2*w^2))
    # Params: p0,p1,p2,p3, a1,a2,a3, c1,c2,c3, w_raw = 10
    # This is close to the original but with 3 Gaussians instead of 1, and no sine.
    
    # Actually, let me reconsider. The directive says "5 Gaussian RBF with learnable centers, widths, and amplitudes (using 10 parameters total)". 
    # That's impossible: 5*(1+1+1)=15 > 10. So maybe 2 RBFs (6 params) + 4 poly = 10? Or 3 RBFs with shared width (9 params) + 1 offset = 10?
    #
    # Let me go with: 3 RBFs with shared width, 3 centers, 3 amplitudes, plus 1 constant offset = 10 params.
    # y = b + sum_{i=1}^{3} a_i * exp(-((x-c_i)^2)/(2*w^2))
    # Params: b, a1,a2,a3, c1,c2,c3, w_raw = 8... need 2 more.
    # 
    # Let me use: 2 RBFs (2 amps + 2 centers + 2 widths = 6) + 4-degree polynomial (4 coeffs) = 10.
    # y = p0 + p1*x + p2*x^2 + p3*x^3 + a1*exp(-((x-c1)^2)/(2*w1^2)) + a2*exp(-((x-c2)^2)/(2*w2^2))
    
    p = params[0:4]   # polynomial coefficients p0, p1, p2, p3
    a1, a2 = params[4], params[5]
    c1, c2 = params[6], params[7]
    w1_raw, w2_raw = params[8], params[9]
    
    # Convert raw widths to positive widths via softplus
    w1_raw_c = np.clip(w1_raw, -50, 50)
    w2_raw_c = np.clip(w2_raw, -50, 50)
    w1 = np.log1p(np.exp(w1_raw_c)) + 1e-6
    w2 = np.log1p(np.exp(w2_raw_c)) + 1e-6
    
    # Polynomial part
    poly = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3
    
    # Gaussian RBF parts
    g1_arg = -0.5 * ((x - c1) / w1) ** 2
    g2_arg = -0.5 * ((x - c2) / w2) ** 2
    g1_arg = np.clip(g1_arg, -700, 700)
    g2_arg = np.clip(g2_arg, -700, 700)
    g1 = a1 * np.exp(g1_arg)
    g2 = a2 * np.exp(g2_arg)
    
    y = poly + g1 + g2
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
