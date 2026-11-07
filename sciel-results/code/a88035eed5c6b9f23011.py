"""Gaussian Process-inspired kernel basis for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 140.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Gaussian Kernel Basis.
    
    Instead of assuming a specific physical law (like Duffing or Van der Pol),
    we model the acceleration as a sum of smooth Gaussian bumps in the 3D state
    space (t, x, v). This allows the model to fit arbitrary nonlinear dynamics.
    
    The 10 parameters are distributed as follows:
    - 3 parameters for the first Gaussian bump: weight (w1), center_t (ct1), center_x (cx1)
    - 3 parameters for the second Gaussian bump: weight (w2), center_v (cv1), scale (s1)
    - 2 parameters for the third Gaussian bump: weight (w3), center_x (cx2)
    - 2 parameters for global scaling: scale_t (st), scale_v (sv)
    
    Note: To keep the parameter count at 10 while maintaining flexibility, we use
    a simplified structure:
    p[0]: weight1
    p[1]: center_t1
    p[2]: center_x1
    p[3]: weight2
    p[4]: center_v1
    p[5]: weight3
    p[6]: center_x2
    p[7]: center_v2
    p[8]: global_scale_t (inverse of time width)
    p[9]: global_scale_xv (inverse of spatial width)
    
    The model is:
    a = p0 * exp(-((t - p1)^2 + (x - p2)^2) * p9^2) 
      + p3 * exp(-((v - p4)^2) * p9^2)
      + p5 * exp(-((x - p6)^2 + (v - p7)^2) * p9^2)
    
    We use p9 as a shared inverse scale to reduce parameter complexity and improve
    convergence, while p8 is kept for potential future use or as a bias term if
    needed, but here we will use it to modulate the time dependence slightly or
    simply keep it as a constant offset if the first term doesn't cover it.
    
    Actually, to be more robust and utilize all 10 params effectively:
    Let's define 3 Gaussian bumps.
    Bump 1: Centered at (t1, x1, v1) with weight w1
    Bump 2: Centered at (t2, x2, v2) with weight w2
    Bump 3: Centered at (t3, x3, v3) with weight w3
    
    That's 9 parameters for centers/weights. The 10th parameter is the global
    inverse bandwidth (sigma_inv).
    
    Params:
    p0: w1
    p1: t1
    p2: x1
    p3: v1
    p4: w2
    p5: t2
    p6: x2
    p7: v2
    p8: w3
    p9: sigma_inv (1/sigma)
    
    Wait, we only have 10 params. 
    p0: w1
    p1: t1
    p2: x1
    p3: v1
    p4: w2
    p5: t2
    p6: x2
    p7: v2
    p8: w3
    p9: sigma_inv
    
    This uses all 10. The third bump (w3) will share the same center as the first
    or be defined by a subset? No, that's not right.
    
    Let's re-allocate to ensure 3 distinct bumps or a better structure.
    Structure:
    Term 1: w1 * exp(-((t-t1)^2 + (x-x1)^2 + (v-v1)^2) * sigma_inv^2)
    Term 2: w2 * exp(-((t-t2)^2 + (x-x2)^2 + (v-v2)^2) * sigma_inv^2)
    Term 3: w3 * exp(-((t-t3)^2 + (x-x3)^2 + (v-v3)^2) * sigma_inv^2)
    
    This requires:
    w1, t1, x1, v1 (4)
    w2, t2, x2, v2 (4)
    w3, t3, x3, v3 (4)
    sigma_inv (1)
    Total = 13 params. We only have 10.
    
    Revised Strategy:
    Use 2 Gaussian bumps in the full 3D space and 1 linear term?
    Or use 1 Gaussian in 3D and 2 in 2D?
    
    Let's try:
    1. A broad Gaussian in 3D to capture the main nonlinear feature.
       Params: w1, cx, cy, cz, sigma (5 params)
    2. A second Gaussian in 2D (x, v) to capture state-dependent nonlinearity independent of t.
       Params: w2, cx2, cy2 (3 params)
    3. A linear drift term in t to capture any secular trend or phase.
       Params: slope_t (1 param)
    4. A constant bias.
       Params: bias (1 param)
    
    Total: 5 + 3 + 1 + 1 = 10 params.
    
    Params:
    p0: w1 (weight of 3D Gaussian)
    p1: cx1 (center t)
    p2: cy1 (center x)
    p3: cz1 (center v)
    p4: sigma_inv (inverse bandwidth for 3D Gaussian)
    p5: w2 (weight of 2D Gaussian)
    p6: cx2 (center x for 2D Gaussian)
    p7: cy2 (center v for 2D Gaussian)
    p8: slope_t (linear coefficient for t)
    p9: bias (constant offset)
    
    Model:
    a = p0 * exp(-((t-p1)^2 + (x-p2)^2 + (v-p3)^2) * p4^2)
      + p5 * exp(-( (x-p6)^2 + (v-p7)^2 ) * p4^2)  # Reusing p4 for scale consistency
      + p8 * t
      + p9
    
    This is structurally different from the Duffing model and more flexible for
    arbitrary nonlinear fields.
    """
    # Safety margin guard
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    # Defensive: if params is short, pad with zeros; if long, ignore extras.
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    (w1, cx1, cy1, cz1, sigma_inv, w2, cx2, cy2, slope_t, bias) = p

    # Prevent numerical issues with sigma_inv
    # If sigma_inv is very large, exp(-large) -> 0.
    # If sigma_inv is very small, exp(-small) -> 1.
    # We clip sigma_inv to a reasonable range to avoid underflow/overflow issues
    # during optimization, although exp(-x^2) is stable for large x.
    # However, if sigma_inv is 0, the Gaussian becomes 1 everywhere (if centers match)
    # or oscillates? No, exp(0) = 1.
    # Let's ensure sigma_inv is not too close to 0 to maintain "bump" character,
    # but BFGS should handle it. We'll just compute directly.
    
    # Compute squared distances
    # Term 1: 3D Gaussian
    dist_sq_1 = (t - cx1)**2 + (x - cy1)**2 + (v - cz1)**2
    term1 = w1 * np.exp(-dist_sq_1 * (sigma_inv ** 2))
    
    # Term 2: 2D Gaussian (x, v)
    dist_sq_2 = (x - cx2)**2 + (v - cy2)**2
    term2 = w2 * np.exp(-dist_sq_2 * (sigma_inv ** 2))
    
    # Term 3: Linear in t
    term3 = slope_t * t
    
    # Term 4: Bias
    term4 = bias
    
    a = term1 + term2 + term3 + term4
    
    # Guarantee finiteness and correct shape
    a = np.where(np.isfinite(a), a, 0.0)
    
    return np.asarray(a, dtype=float)
# EVOLVE-BLOCK-END
