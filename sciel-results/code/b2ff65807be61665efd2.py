"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Duffing-van der Pol-inspired
    basis with input scaling and tanh saturation for numerical conditioning.
    
    Basis (10 terms):
      1. constant
      2. x_s  (scaled position)
      3. v_s  (scaled velocity)
      4. x_s^2
      5. v_s^2
      6. x_s * v_s
      7. tanh(x_s)
      8. tanh(v_s)
      9. x_s^3
      10. v_s^3
    """
    # Scale inputs to approximately [-1, 1] for better conditioning
    x_s = x / 0.5728
    v_s = v / 0.2821

    # Precompute basis terms
    t0 = np.ones_like(x_s)
    t1 = x_s
    t2 = v_s
    t3 = x_s * x_s
    t4 = v_s * v_s
    t5 = x_s * v_s
    t6 = np.tanh(x_s)
    t7 = np.tanh(v_s)
    t8 = x_s * x_s * x_s
    t9 = v_s * v_s * v_s

    # Linear combination with 10 parameters
    return (params[0] * t0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9] * t9)
# EVOLVE-BLOCK-END
