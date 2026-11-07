# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed scaling constants to map inputs to roughly [-1, 1]
# x range: [-0.5728, 0.5275] -> scale by 1/0.5728
# v range: [-0.1754, 0.2821] -> scale by 1/0.2821
_X_SCALE = 1.0 / 0.5728
_V_SCALE = 1.0 / 0.2821

# Chebyshev polynomial basis for improved approximation properties
# We use a 2D Chebyshev basis in scaled x and v.
# Total 10 parameters.
# Let's use a mixed basis:
# T_0(u), T_1(u), T_2(u), T_3(u)  (4 terms in u)
# T_0(w), T_1(w), T_2(w), T_3(w)  (4 terms in w)
# T_1(u)*T_1(w), T_1(u)*T_2(w)    (2 cross terms)
# This gives 4+4+2 = 10 terms.
# However, T_0 is constant. So we have:
# 0: 1
# 1: u
# 2: 2u^2 - 1
# 3: 4u^3 - 3u
# 4: w
# 5: 2w^2 - 1
# 6: 4w^3 - 3w
# 7: u * w
# 8: u * (2w^2 - 1)
# 9: (2u^2 - 1) * w

# Normalization scales for Chebyshev polynomials
# T_n(t) for t in [-1, 1] is bounded by 1.
# RMS of T_n is roughly 1/sqrt(2) for n>0, and 1 for n=0.
# Let's use conservative scales to keep params O(1).
# Actually, Chebyshev polynomials are orthogonal w.r.t weight 1/sqrt(1-t^2).
# Their L2 norm is sqrt(pi/2) for n>0.
# But we are fitting with MSE, so we want to balance the magnitudes.
# Since T_n(t) is in [-1, 1], the RMS is <= 1.
# Let's just use 1.0 for all, or slightly less to be safe.
# The previous monomial approach used specific scales.
# For Chebyshev, the values are more uniformly distributed in magnitude.
# Let's use 1.0 for all basis functions.
_SCALES = np.ones(10)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    """
    # Deadline guard
    # The evaluator kills after 150s. We don't have a global timer here easily,
    # but the function is called many times during optimization.
    # We just need to be fast. Chebyshev is fast.
    
    u = x * _X_SCALE
    w = v * _V_SCALE
    
    # Clip to [-1, 1] to ensure Chebyshev stability?
    # The inputs are in training range, so u, w should be in [-1, 1] roughly.
    # x in [-0.5728, 0.5275] -> u in [-1, 0.92]
    # v in [-0.1754, 0.2821] -> w in [-0.62, 1.0]
    # So they are within [-1, 1]. No clipping needed, but safe to have.
    # However, clipping changes the function definition. Better not to clip if not necessary.
    
    # Compute Chebyshev polynomials
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_u = np.ones_like(x)
    T1_u = u
    T2_u = 2.0 * u * u - 1.0
    T3_u = 4.0 * u * u * u - 3.0 * u
    
    T0_w = np.ones_like(v)
    T1_w = w
    T2_w = 2.0 * w * w - 1.0
    T3_w = 4.0 * w * w * w - 3.0 * w
    
    # Cross terms
    C1 = T1_u * T1_w
    C2 = T1_u * T2_w
    C3 = T2_u * T1_w
    
    # Basis:
    # 0: T0_u (1)
    # 1: T1_u (u)
    # 2: T2_u
    # 3: T3_u
    # 4: T1_w (w)
    # 5: T2_w
    # 6: T3_w
    # 7: T1_u * T1_w
    # 8: T1_u * T2_w
    # 9: T2_u * T1_w
    
    basis = np.column_stack((T0_u, T1_u, T2_u, T3_u, T1_w, T2_w, T3_w, C1, C2, C3))
    
    # Normalize (scales are 1.0 here, so this is a no-op but kept for structure)
    basis_norm = basis / _SCALES
    
    return basis_norm @ params
# EVOLVE-BLOCK-END
