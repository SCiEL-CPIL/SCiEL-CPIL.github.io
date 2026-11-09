"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model with 10 parameters.
    
    Structural change: Hybrid non-linear parametric model.
    Combines the rich linear basis of the current program (polynomials, tanh, exp)
    with the structural flexibility of the best inspiration (Gaussian, Sine, Scaled Tanh).
    
    Parameters are split:
    - p[0..4]: Amplitudes for polynomial/tanh/exp basis (linear in params)
    - p[5..9]: Shape parameters for a Gaussian and Sine term (non-linear in params)
    
    This allows the model to fit local features (Gaussian) and periodic features (Sine)
    while maintaining global stability via the polynomial/tanh basis.
    """
    # Deadline guard: Track elapsed time to ensure we finish within limits.
    # Since the evaluator calls this function repeatedly during BFGS,
    # we must be fast. The guard here is a safety net for pathological cases,
    # but the primary strategy is efficient vectorized operations.
    start_time = time.time()
    
    p = params
    x_safe = np.clip(x, -10, 10)
    
    # --- Part 1: Linear Basis (Stable, captures global trend) ---
    # Derived from current program's best features
    x2 = x_safe * x_safe
    x3 = x2 * x_safe
    tanh_x = np.tanh(x_safe)
    exp_neg_x2 = np.exp(-x2)
    
    b0 = x_safe          # linear
    b1 = x2              # quadratic
    b2 = x3              # cubic
    b3 = tanh_x          # tanh
    b4 = x_safe * exp_neg_x2  # x * exp(-x^2)
    
    # --- Part 2: Non-Linear Structural Terms (Captures local/periodic features) ---
    # Derived from Inspiration 3 (score 454356), which outperformed the pure linear basis.
    
    # Gaussian Term: p[5] * exp(-exp(p[6]) * (x - p[7])^2)
    # p[5]: amplitude
    # p[6]: log-width (exp(p[6]) is width)
    # p[7]: center
    p5 = p[5]
    p6 = np.clip(p[6], -20, 20)  # Prevent overflow in exp
    p7 = p[7]
    width = np.exp(p6)
    gauss = p5 * np.exp(-width * (x_safe - p7) ** 2)
    
    # Sine Term: p[8] * sin(p[9] * x)
    # p[8]: amplitude
    # p[9]: frequency
    p8 = p[8]
    p9 = p[9]
    sine = p8 * np.sin(p9 * x_safe)
    
    # Combine
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        gauss +
        sine
    )
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline (unlikely to be hit in vectorized ops, but safe)
    elapsed = time.time() - start_time
    if elapsed > 110.0:
        # If we are taking too long, return a simple fallback to avoid timeout
        # This is a safety net; normally vectorized ops are <1ms for N=200
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
