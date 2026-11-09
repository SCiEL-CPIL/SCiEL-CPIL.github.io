"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Chebyshev model: f(x) = P(x) / Q(x) where
    P(x) = sum_{i=0}^{4} a_i T_i(t),  Q(x) = 1 + sum_{j=1}^{4} b_j T_j(t)
    with t = x / 2 to map [-2, 2] -> [-1, 1].
    
    This uses 9 free parameters (a_0..a_4, b_1..b_4) plus one extra parameter
    for scaling. The denominator is constrained to be positive via tanh scaling
    to prevent division by zero and BFGS divergence.
    """
    t = x / 2.0  # scale to [-1, 1]
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    
    # Numerator: P(x) = a0*T0 + a1*T1 + a2*T2 + a3*T3 + a4*T4
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    numerator = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: Q(x) = 1 + b1*T1 + b2*T2 + b3*T3 + b4*T4
    # Constrain b_j to be small to keep denominator well-behaved
    # Use tanh to bound the perturbation: 1 + 0.5*tanh(b_j) ensures Q stays in [0.5, 1.5]
    b1 = 0.5 * np.tanh(params[5])
    b2 = 0.5 * np.tanh(params[6])
    b3 = 0.5 * np.tanh(params[7])
    b4 = 0.5 * np.tanh(params[8])
    
    denominator = 1.0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    
    # Ensure denominator is never too close to zero
    # The tanh constraint keeps |b_j| < 0.5, and |T_j(t)| <= 1 for t in [-1,1]
    # So |sum b_j T_j| < 4 * 0.5 = 2, meaning Q(x) in (-1, 3)
    # But to be safe, clip the denominator away from zero
    denominator = np.where(np.abs(denominator) < 0.1, 0.1 * np.sign(denominator), denominator)
    # Handle the case where denominator is exactly zero (shouldn't happen with clipping)
    denominator = np.where(denominator == 0, 0.1, denominator)
    
    # Scale parameter for overall amplitude
    scale = params[9]
    
    result = scale * numerator / denominator
    
    return result
# EVOLVE-BLOCK-END
