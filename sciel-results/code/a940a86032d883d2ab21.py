"""
Strategy: Bounded Sigmoidal Basis with Physics-Informed Grouping

Instead of using unbounded polynomials (which can lead to ill-conditioning and 
poor generalization) or local RBFs (which require careful center placement), 
this strategy uses a global basis of bounded, smooth non-linear functions (tanh).

The basis is structured to respect the physics of a nonlinear oscillator:
1. Restoring forces (functions of x): linear, cubic-like (tanh), quintic-like.
2. Damping/Injection forces (functions of v and x): linear, quadratic-like, 
   cross-coupled (Van der Pol term x^2*v).

Using tanh instead of power series (x^3, x^5) ensures that the gradients 
remain well-behaved during BFGS optimization, as tanh' is bounded and 
smooth. This often allows the optimizer to find a better global minimum 
compared to the steep valleys of high-order polynomial MSE landscapes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a bounded sigmoidal basis.
    
    Inputs:
    x: Displacement (range approx [-0.57, 0.53])
    v: Velocity (range approx [-0.18, 0.28])
    params: Array of 10 fitted constants.
    
    Returns:
    Acceleration prediction.
    """
    # Normalize inputs to standard scale to improve BFGS conditioning.
    # x center ~0, scale ~0.55
    # v center ~0, scale ~0.25
    x_n = x / 0.55
    v_n = v / 0.25
    
    # Pre-compute basis functions.
    # We use tanh to model non-linearities. tanh(z) ~ z for small z, 
    # saturates to +/- 1 for large z. This mimics soft saturation 
    # common in physical oscillators.
    
    # Group 1: Restoring Force (depends primarily on x)
    # t0: Bias (constant acceleration offset)
    t0 = np.ones_like(x_n)
    
    # t1: Linear restoring force (Hooke's law)
    t1 = x_n
    
    # t2: Non-linear restoring (Duffing-like). 
    # Using tanh(x^3) or similar to capture cubic behavior but bounded.
    # Actually, tanh(x_n * 3) is a good proxy for a soft cubic.
    t2 = np.tanh(x_n * 3.0)
    
    # t3: Higher order restoring (quintic-like saturation)
    t3 = np.tanh(x_n * 6.0)
    
    # Group 2: Damping and Injection (depends primarily on v, modulated by x)
    # t4: Linear damping (viscous)
    t4 = v_n
    
    # t5: Non-linear damping (cubic-like, bounded)
    # tanh(v^3) is not standard, let's use tanh(v * 3) for cubic-like slope change
    t5 = np.tanh(v_n * 3.0)
    
    # t6: Quadratic damping/injection (energy flow term)
    # v^2 is always positive, representing energy dissipation or injection 
    # that depends on speed squared.
    t6 = v_n ** 2
    
    # Group 3: Cross-coupling terms (Parametric / Van der Pol specific)
    # The Van der Pol equation has a term -mu * x^2 * v.
    # We model this with a bounded version to prevent explosion.
    
    # t7: Linear cross term x * v (parametric coupling)
    t7 = x_n * v_n
    
    # t8: Van der Pol specific term: x^2 * v
    # We use tanh(x^2 * 3) * v to bound the x-dependence.
    t8 = np.tanh(x_n ** 2 * 3.0) * v_n
    
    # t9: Higher order cross term or velocity saturation cross term
    # Let's add a term that captures velocity-dependent stiffness or 
    # displacement-dependent damping asymmetry.
    # tanh(v) * x is similar to t7 but bounded. 
    # Let's use a term that captures the asymmetry often found in 
    # nonlinear oscillators: x * tanh(v)
    t9 = x_n * np.tanh(v_n * 2.0)
    
    # Combine with parameters
    # params is passed as a 1D array of length 10
    p = params
    
    out = (p[0] * t0 + 
           p[1] * t1 + 
           p[2] * t2 + 
           p[3] * t3 + 
           p[4] * t4 + 
           p[5] * t5 + 
           p[6] * t6 + 
           p[7] * t7 + 
           p[8] * t8 + 
           p[9] * t9)
           
    return out
# EVOLVE-BLOCK-END
