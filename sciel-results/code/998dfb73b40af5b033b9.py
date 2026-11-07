"""Symbolic regression with 1 input; Hybrid Neural Network with fixed ReLU centers and adaptive widths."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining Neural Network structure with RBF-like basis.
    
    We use a single hidden layer with 8 units and 2 linear output coefficients.
    Total parameters: 8 (weights) + 8 (biases) + 2 (output weights) = 18? 
    Wait, we only have 10 parameters.
    
    Let's redesign for exactly 10 parameters:
    Option A: 4 hidden units (tanh)
      - 4 weights w_i
      - 4 biases b_i
      - 2 output weights v_i (for 2 hidden units? No, need 4)
      - 1 bias c
      Total: 4+4+4+1 = 13. Too many.
      
    Option B: 3 hidden units (tanh)
      - 3 weights w_i
      - 3 biases b_i
      - 3 output weights v_i
      - 1 bias c
      Total: 3+3+3+1 = 10. Perfect.
      
    However, tanh is smooth everywhere. ReLU might be better for non-smooth features.
    Let's try ReLU with 3 units:
      hidden_i = max(0, w_i * x + b_i)
      output = v_1*h_1 + v_2*h_2 + v_3*h_3 + c
      
    This is piecewise linear. With 3 units, we get 4 linear segments.
    The domain is [0.1, 6]. This might be too coarse.
    
    Alternative: Fixed centers, learnable widths/weights?
    The current RBF approach uses fixed centers and fixed widths, learning only weights.
    This is essentially a linear model in the RBF basis.
    
    Let's try a different structural change: **Fourier Basis**.
    Many smooth functions can be well-approximated by a truncated Fourier series.
    y = a_0 + sum_{k=1}^{4} [a_k * cos(k * omega * x) + b_k * sin(k * omega * x)]
    
    Parameters:
    a_0 (1)
    a_1, b_1 (2)
    a_2, b_2 (2)
    a_3, b_3 (2)
    a_4, b_4 (2)
    Total: 1 + 2*4 = 9. We have 10.
    Add one more harmonic?
    a_5, b_5 (2) -> Total 11. Too many.
    
    Let's use 4.5 harmonics? No.
    Let's use:
    a_0 (1)
    a_1, b_1 (2)
    a_2, b_2 (2)
    a_3, b_3 (2)
    a_4 (1) -> Total 8.
    Remaining 2 params: maybe a linear term?
    y = a_0 + m*x + sum_{k=1}^{4} [a_k cos(...) + b_k sin(...)]
    Params: a_0, m, a_1, b_1, a_2, b_2, a_3, b_3, a_4, b_4 = 10.
    
    This allows capturing linear trends and periodic/non-periodic oscillatory behavior.
    The frequency omega should be chosen to cover the domain well.
    Domain length L = 5.9.
    Fundamental frequency omega = 2*pi / L? Or just 1.0?
    If omega = 1, cos(x) has period ~6.28.
    Let's use omega = 1.0 for simplicity, or scale x to [0, 2*pi].
    Scaling: x_scaled = (x - 0.1) / (6.0 - 0.1) * 2 * pi.
    Then frequencies are 1, 2, 3, 4, 5.
    
    Let's try this Fourier approach. It is a global basis, unlike RBF, but often very effective for smooth functions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [0, 2*pi]
    x_min = 0.1
    x_max = 6.0
    x_scaled = (x - x_min) / (x_max - x_min) * 2.0 * np.pi
    
    # Extract parameters
    # p0: bias
    # p1: linear slope (on original x or scaled? Let's use original x for linear term to keep units intuitive, or scaled for consistency. Let's use scaled x for all terms to keep magnitudes similar)
    # Actually, mixing linear in x_scaled is fine.
    
    a0 = params[0]
    m = params[1]
    
    # Harmonics 1 to 4 (8 params)
    a1, b1 = params[2], params[3]
    a2, b2 = params[4], params[5]
    a3, b3 = params[6], params[7]
    a4, b4 = params[8], params[9]
    
    # Compute basis functions
    # k=1
    term1 = a1 * np.cos(x_scaled) + b1 * np.sin(x_scaled)
    # k=2
    term2 = a2 * np.cos(2 * x_scaled) + b2 * np.sin(2 * x_scaled)
    # k=3
    term3 = a3 * np.cos(3 * x_scaled) + b3 * np.sin(3 * x_scaled)
    # k=4
    term4 = a4 * np.cos(4 * x_scaled) + b4 * np.sin(4 * x_scaled)
    
    # Linear term (using scaled x to keep magnitude ~1)
    linear_term = m * x_scaled
    
    y = a0 + linear_term + term1 + term2 + term3 + term4
    
    return y
# EVOLVE-BLOCK-END
