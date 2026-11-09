"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + Linear model.
    
    Params layout (length 10):
    - params[0], params[1]: Linear slope and intercept (global trend)
    - params[2], params[3], params[4]: Centers (c1, c2, c3) of 3 Gaussians
    - params[5], params[6], params[7]: Widths (w1, w2, w3) of 3 Gaussians (positive via tanh)
    - params[8], params[9], params[10]: Weights (a1, a2, a3) of 3 Gaussians
    
    Note: params is length 10, so indices 0..9.
    Let's map:
    0: slope
    1: intercept
    2: center 1
    3: center 2
    4: center 3
    5: width 1 (raw, transformed)
    6: width 2 (raw, transformed)
    7: width 3 (raw, transformed)
    8: weight 1
    9: weight 2
    Wait, I need 3 weights. I only have 10 params.
    
    Revised Layout:
    0: slope
    1: intercept
    2: center 1
    3: center 2
    4: width 1 (raw)
    5: width 2 (raw)
    6: weight 1
    7: weight 2
    8: center 3
    9: width 3 (raw)
    
    This leaves weight 3 missing. I can't fit 3 full RBFs with 10 params if I keep linear.
    Let's use 2 RBFs + Linear.
    0: slope
    1: intercept
    2: center 1
    3: width 1 (raw)
    4: weight 1
    5: center 2
    6: width 2 (raw)
    7: weight 2
    8: center 3
    9: weight 3
    Widths for 3rd RBF can be fixed or derived? 
    Better: Use 3 RBFs but share widths? Or just use 2 RBFs and a quadratic term?
    
    Let's stick to 2 RBFs + Linear (slope, intercept) + Quadratic (x^2 coeff).
    0: slope (x)
    1: intercept
    2: quad coeff (x^2)
    3: center 1
    4: width 1 (raw)
    5: weight 1
    6: center 2
    7: width 2 (raw)
    8: weight 2
    9: center 3 (maybe unused or used for a 3rd narrow spike with fixed width?)
    
    Actually, let's use a "Mixture of Experts" style or just a flexible sum of Gaussians.
    If I use 3 Gaussians, I need 3 centers, 3 widths, 3 weights = 9 params.
    Plus 1 global offset = 10 params. No linear term?
    A global offset is better than a linear term for RBFs often.
    
    Layout:
    0: offset
    1: center 1
    2: width 1 (raw)
    3: weight 1
    4: center 2
    5: width 2 (raw)
    6: weight 2
    7: center 3
    8: width 3 (raw)
    9: weight 3
    
    This is a pure RBF sum with offset. It can approximate any smooth function well if centers move.
    However, if the function is linear, RBFs might struggle to fit a perfect line without large widths.
    Large widths -> approximates a parabola/constant.
    
    Let's try a mix: 1 Linear + 2 RBFs.
    0: slope
    1: intercept
    2: center 1
    3: width 1 (raw)
    4: weight 1
    5: center 2
    6: width 2 (raw)
    7: weight 2
    8: center 3
    9: weight 3 (width 3 fixed to 0.5?)
    
    Let's go with 3 RBFs and 1 offset. It's a standard universal approximator setup.
    The BFGS optimizer should be able to move the centers.
    To ensure stability, widths are transformed via exp(raw) or abs(raw)+epsilon.
    Centers are raw.
    Weights are raw.
    """
    # Extract parameters
    offset = params[0]
    c1, w1_raw, a1 = params[1], params[2], params[3]
    c2, w2_raw, a2 = params[4], params[5], params[6]
    c3, w3_raw, a3 = params[7], params[8], params[9]
    
    # Transform widths to be positive and avoid division by zero/explosion
    # Using exp ensures positive width. If raw is large, width is large (flat).
    # If raw is small/negative, width is small (spiky).
    # To prevent underflow/overflow in exp, clip raw inputs.
    w1 = np.exp(np.clip(w1_raw, -5, 5))
    w2 = np.exp(np.clip(w2_raw, -5, 5))
    w3 = np.exp(np.clip(w3_raw, -5, 5))
    
    # Compute Gaussians
    # g(x; c, w) = exp(-(x-c)^2 / (2*w^2))
    # Note: w is the standard deviation.
    g1 = np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    g2 = np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    g3 = np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    # Combine
    y = offset + a1 * g1 + a2 * g2 + a3 * g3
    
    return y
# EVOLVE-BLOCK-END
