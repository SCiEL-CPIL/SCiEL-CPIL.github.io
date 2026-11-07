import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # 10s safety margin before 150s kill

def equation(b, s, temp, pH, params):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    p = params
    
    # Normalization to roughly [0, 1] based on training ranges
    # b: [1, 5] -> [0.2, 1.0]
    # s: [1, 5] -> [0.2, 1.0]
    # temp: [1, 40] -> [0, 1]
    # pH: [1, 10] -> [0, 1]
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 1.0) / 9.0
    
    # Structural Change: 
    # Instead of a multiplicative model (Monod * Gaussian * Gaussian), 
    # we use a single, unconstrained 4th-order multivariate polynomial 
    # in the normalized inputs.
    #
    # This abandons the assumption of independent factorization.
    # The 10 parameters are distributed across different orders of the polynomial:
    #
    # Order 0 (Constant): p[0]
    # Order 1 (Linear):   p[1]*b_n + p[2]*s_n + p[3]*t_n + p[4]*ph_n
    # Order 2 (Quadratic):
    #   p[5] * (b_n^2 + s_n^2 + t_n^2 + ph_n^2)  [Isotropic quadratic]
    #   p[6] * (b_n*s_n + t_n*ph_n)              [Key interactions]
    # Order 4 (Quartic):
    #   p[7] * (b_n^4 + s_n^4 + t_n^4 + ph_n^4)  [Isotropic quartic]
    #   p[8] * (b_n^2*s_n^2 + t_n^2*ph_n^2)      [Key quadratic interactions]
    #
    # Remaining parameter:
    #   p[9] * (b_n*s_n*t_n*ph_n)                [Full 4-way interaction]
    #
    # This structure allows the optimizer to fit arbitrary high-order interaction 
    # terms and non-monotonic dependencies across all variables simultaneously,
    # which is particularly useful if the underlying growth rate function has 
    # complex, coupled dependencies (e.g., temperature and pH effects are not 
    # perfectly separable, or substrate saturation interacts with biomass).
    
    # Precompute powers
    b2 = b_n * b_n
    s2 = s_n * s_n
    t2 = t_n * t_n
    ph2 = ph_n * ph_n
    
    b4 = b2 * b2
    s4 = s2 * s2
    t4 = t2 * t2
    ph4 = ph2 * ph2
    
    # Order 0
    term0 = p[0]
    
    # Order 1
    term1 = p[1] * b_n + p[2] * s_n + p[3] * t_n + p[4] * ph_n
    
    # Order 2
    # Isotropic quadratic
    quad_iso = b2 + s2 + t2 + ph2
    # Key interactions
    quad_inter = b_n * s_n + t_n * ph_n
    term2 = p[5] * quad_iso + p[6] * quad_inter
    
    # Order 4
    # Isotropic quartic
    quart_iso = b4 + s4 + t4 + ph4
    # Key quadratic interactions
    quart_inter = b2 * s2 + t2 * ph2
    term4 = p[7] * quart_iso + p[8] * quart_inter
    
    # Full 4-way interaction (Order 4)
    term4_full = p[9] * (b_n * s_n * t_n * ph_n)
    
    # Sum all terms
    result = term0 + term1 + term2 + term4 + term4_full
    
    return result
