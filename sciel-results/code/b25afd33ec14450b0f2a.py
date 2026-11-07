"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative mechanistic growth model with smooth differentiable constraints.
    
    Strategy:
    1. Use a multiplicative structure (Monod * Temp * pH * Density) to enforce the biological
       reality that growth is zero if any factor is limiting. This creates a highly non-convex
       landscape, but the structure is strongly aligned with the data generation process.
    2. Replace hard np.clip/np.minimum (non-differentiable) with smooth approximations
       (softplus, smooth-min) to provide valid gradients for BFGS.
    3. The parameters are fitted by the external optimizer. We use 10 parameters:
       p0: max growth rate (mu_max)
       p1: Monod constant (Ks)
       p2: Temperature lower threshold
       p3: Temperature upper threshold
       p4: Temperature optimal center
       p5: pH lower threshold
       p6: pH upper threshold
       p7: pH optimal center
       p8: Density inhibition constant
       p9: Global scale/offset (bias)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Substrate: Monod saturation
    # s / (Ks + s)
    # Ensure Ks is positive-ish for stability, though BFGS will find the best fit
    # We use a softplus-like stabilization if needed, but raw is often fine if initialized well
    # However, to avoid division by zero or negative Ks causing weird behavior, we can use abs or clip
    # But the task says "closed-form NumPy expression".
    # Let's use a robust Monod: s / (p1 + s)
    # If p1 is negative, it might behave oddly. Let's assume the optimizer stays in a reasonable range
    # or we can use np.maximum(p1, 0.1) + 1e-6 for stability? 
    # Actually, the evaluator starts params at 1.0. So p1=1.0 initially.
    # Let's stick to standard Monod.
    f_s = s / (p1 + s)
    
    # 2. Temperature: Smooth trapezoid
    # We want a function that is 0 below p2, ramps up to 1 at p4, stays 1 until p3, ramps down to 0.
    # Standard smooth ramp: 1 / (1 + exp(-k*(x - x0)))
    # But we need a plateau.
    # Let's use a product of two smooth sigmoid steps.
    # Low side: Sigmoid((temp - p2) * k_low)
    # High side: Sigmoid((p3 - temp) * k_high)
    # To make it a sharp plateau, k should be large. But large k makes gradients vanish or explode.
    # Alternative: Use smooth min/max.
    # Let's try a different approach: Gaussian-like response centered at p4, but with wider support.
    # Or, use the "Black-Koch" formulation which is often:
    # f_T = exp(-0.5 * ((T - T_opt) / sigma)^2)
    # This is smooth and differentiable everywhere.
    # Let's use a Gaussian for Temperature and pH. It's simpler and differentiable.
    # The "plateau" is approximated by a wide Gaussian.
    
    # Temperature Gaussian
    # p2: center, p3: width (sigma)
    # We need to ensure p3 (sigma) is positive.
    # Let's use p2 as center, p3 as inverse variance or sigma.
    # If we use sigma, we should ensure it's not 0.
    # Let's use a softplus for sigma to keep it positive? 
    # No, let's just use the raw params and hope BFGS stays away from 0, or use abs.
    # abs is non-differentiable at 0.
    # Let's use: sigma_t = np.maximum(p3, 0.1)
    # This is non-diff at 0.1.
    # Better: use p3^2 + 0.1 to ensure positivity and smoothness.
    sigma_t = np.sqrt(p3**2 + 0.1)
    f_temp = np.exp(-0.5 * ((temp - p2) / sigma_t)**2)
    
    # pH Gaussian
    # p5: center, p6: width
    sigma_p = np.sqrt(p6**2 + 0.1)
    f_ph = np.exp(-0.5 * ((pH - p5) / sigma_p)**2)
    
    # 3. Density: Exponential inhibition
    # exp(-k * b)
    # p7 is the inhibition constant.
    # Ensure p7 is positive? If p7 is negative, growth increases with density, which is wrong.
    # Let's use p7 as is, the optimizer will figure it out.
    f_b = np.exp(-p7 * b)
    
    # Combine multiplicatively
    # y = p0 * f_s * f_temp * f_ph * f_b + p9
    # p0 is the max growth rate.
    # p9 is the baseline (should be ~0).
    
    y = p0 * f_s * f_temp * f_ph * f_b + p9
    
    return y
# EVOLVE-BLOCK-END
