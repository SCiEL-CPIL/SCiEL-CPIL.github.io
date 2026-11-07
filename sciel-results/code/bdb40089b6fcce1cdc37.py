"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Convex, linear-in-parameters basis expansion.
    
    All terms are fixed functions of the inputs; the only free variables are the
    coefficients p0..p9 which appear linearly. This makes the MSE a convex
    quadratic in params, so BFGS (or any least-squares solver) finds the global
    optimum from the all-ones start.
    
    Basis design (scaled inputs for numerical conditioning):
      u_b  = (b - 1) / 4        in [0, 1]
      u_s  = (s - 1) / 4        in [0, 1]
      u_t  = (temp - 1) / 39    in [0, 1]
      u_p  = (pH - 2) / 10      in [0, 1]
    
    Terms:
      0: constant (bias)
      1: u_b                      linear density
      2: u_s                      linear substrate
      3: u_t                      linear temperature
      4: u_p                      linear pH
      5: Monod saturation s/(1+s)  (bounded [0,1])
      6: RBF near optimal temp/pH  exp(-2*((u_t-0.92)^2 + (u_p-0.5)^2))
      7: RBF near low temp         exp(-2*((u_t-0.1)^2 + (u_p-0.5)^2))
      8: RBF high density/subst    exp(-2*((u_b-1)^2 + (u_s-1)^2 + (u_t-0.92)^2))
      9: RBF low temp/pH stress    exp(-2*((u_t-0.05)^2 + (u_p-0.15)^2))
    """
    # Guard against any pathological params that might cause overflow
    # (not strictly necessary for convex model, but safe)
    p = np.clip(params, -1e6, 1e6)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p

    # Scale inputs to [0, 1]
    u_b = (b - 1.0) / 4.0
    u_s = (s - 1.0) / 4.0
    u_t = (temp - 1.0) / 39.0
    u_p = (pH - 2.0) / 10.0

    # Clip scaled values to prevent extreme values from OOD inputs
    u_b = np.clip(u_b, -1.0, 2.0)
    u_s = np.clip(u_s, -1.0, 2.0)
    u_t = np.clip(u_t, -1.0, 2.0)
    u_p = np.clip(u_p, -1.0, 2.0)

    # Monod saturation (bounded, smooth)
    f_monod = s / (1.0 + s)

    # RBF terms with gamma=2.0, fixed centers
    # Center 1: optimal temp (u_t~0.92), neutral pH (u_p=0.5)
    rbf_opt = np.exp(-2.0 * ((u_t - 0.92)**2 + (u_p - 0.5)**2))

    # Center 2: low temp stress (u_t~0.1), neutral pH
    rbf_lowt = np.exp(-2.0 * ((u_t - 0.1)**2 + (u_p - 0.5)**2))

    # Center 3: high density/substrate, optimal temp
    rbf_hi = np.exp(-2.0 * ((u_b - 1.0)**2 + (u_s - 1.0)**2 + (u_t - 0.92)**2))

    # Center 4: low temp and low pH stress
    rbf_stress = np.exp(-2.0 * ((u_t - 0.05)**2 + (u_p - 0.15)**2))

    y = (p0
         + p1 * u_b
         + p2 * u_s
         + p3 * u_t
         + p4 * u_p
         + p5 * f_monod
         + p6 * rbf_opt
         + p7 * rbf_lowt
         + p8 * rbf_hi
         + p9 * rbf_stress)

    return y
# EVOLVE-BLOCK-END
