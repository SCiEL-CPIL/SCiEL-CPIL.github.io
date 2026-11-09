"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed Duffing-van der Pol structure for nonlinear oscillator acceleration.
    
    a = p1*x + p2*x^3 - p3*v - p4*v^3 + p5*x*v^2 + p6*v^2*x^2 + p7*x^2*v + p8*x^4 + p9*v^4 + p10
    
    Scaling:
    x in [-0.5728, 0.5275] -> xs = x / 0.55 (roughly [-1, 1])
    v in [-0.1754, 0.2821] -> vs = v / 0.25 (roughly [-0.7, 1.1])
    """
    # Scale inputs to approximately [-1, 1] for numerical conditioning
    xs = x / 0.55
    vs = v / 0.25
    
    # Precompute powers
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2
    vs2 = vs * vs
    vs3 = vs2 * vs
    vs4 = vs2 * vs2
    
    # Physics-informed terms
    # Duffing: linear + cubic restoring force
    # Van der Pol: linear + cubic damping
    # Cross terms for nonlinear coupling
    term0 = xs          # p1 * x (linear restoring)
    term1 = xs3         # p2 * x^3 (cubic restoring, Duffing)
    term2 = -vs         # -p3 * v (linear damping)
    term3 = -vs3        # -p4 * v^3 (cubic damping, Van der Pol)
    term4 = xs * vs2    # p5 * x * v^2 (nonlinear coupling)
    term5 = vs2 * xs2   # p6 * v^2 * x^2 (quadratic coupling)
    term6 = xs2 * vs    # p7 * x^2 * v (cross cubic)
    term7 = xs4         # p8 * x^4 (quartic restoring)
    term8 = vs4         # p9 * v^4 (quartic damping)
    term9 = np.ones_like(x)  # p10 (constant/bias)
    
    # Linear combination
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
