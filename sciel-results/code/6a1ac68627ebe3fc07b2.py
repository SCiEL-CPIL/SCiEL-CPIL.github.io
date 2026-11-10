"""Symbolic regression with Gaussian RBF basis + internal refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rbf_predict(x, params):
    """
    Predict using 5 Gaussian RBF basis functions + linear term.
    params layout (length 10):
      [0] = amplitude of RBF 0
      [1] = mean of RBF 0
      [2] = width of RBF 0
      [3] = amplitude of RBF 1
      [4] = mean of RBF 1
      [5] = width of RBF 1
      [6] = amplitude of RBF 2
      [7] = mean of RBF 2
      [8] = width of RBF 2
      [9] = linear slope (multiplied by x)
    
    We use 3 RBFs (9 params) + 1 linear slope = 10 params.
    Actually, let's use 4 RBFs with shared structure? No, 10 params total.
    
    Let's do: 3 RBFs (amp, mean, width each = 9) + 1 constant offset.
    Or: 3 RBFs + linear term (slope*x).
    
    Let me use: 3 RBFs (amp, mean, width) = 9 params, plus params[9] as intercept.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract RBF parameters
    amp0, mean0, width0 = p[0], p[1], p[2]
    amp1, mean1, width1 = p[3], p[4], p[5]
    amp2, mean2, width2 = p[6], p[7], p[8]
    intercept = p[9]
    
    # Ensure widths are positive to avoid numerical issues
    w0 = np.abs(width0) + 1e-6
    w1 = np.abs(width1) + 1e-6
    w2 = np.abs(width2) + 1e-6
    
    # Compute RBFs
    g0 = np.exp(-((x - mean0) ** 2) / (2.0 * w0 ** 2))
    g1 = np.exp(-((x - mean1) ** 2) / (2.0 * w1 ** 2))
    g2 = np.exp(-((x - mean2) ** 2) / (2.0 * w2 ** 2))
    
    return amp0 * g0 + amp1 * g1 + amp2 * g2 + intercept


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Main entry point. Uses a 3-Gaussian RBF + intercept model with 10 params.
    The evaluator will fit params via BFGS from all-ones start.
    """
    return _rbf_predict(x, params)
# EVOLVE-BLOCK-END
