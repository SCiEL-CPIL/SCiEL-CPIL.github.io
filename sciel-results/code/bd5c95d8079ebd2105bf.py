"""Physics-informed nonlinear oscillator regression with multi-start BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Global deadline guard: 150s limit, keep 20s buffer for safety
_DEADLINE = time.time() + 130


def _time_left() -> float:
    return _DEADLINE - time.time()


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction for a nonlinear oscillator.
    
    Model: a = -w^2 * x - gamma * v - beta * x^3 + F0 * cos(w_d * t + phi)
    
    Since t is not an input, we estimate it from the phase of the linear oscillator.
    For a linear oscillator x = A cos(w t + phi), v = -A w sin(w t + phi).
    We can estimate the phase theta = w t + phi via atan2(-v, x) / w.
    However, w is a parameter. To keep it closed-form and differentiable, 
    we use a proxy for the driving term that depends on x and v in a way 
    that captures the oscillatory nature.
    
    Alternative approach: 
    The task asks for equation(x, v, params). 
    If the system is a Duffing oscillator with driving, the driving force is 
    typically F0 * cos(w_d * t + phi). 
    Without t, we can't directly compute cos(w_d * t + phi) unless we estimate t.
    
    Let's assume the motion is dominated by the natural frequency w0 for the 
    position/velocity phase relationship. 
    x ≈ A cos(theta)
    v ≈ -A w0 sin(theta)
    => tan(theta) = -v / (w0 x)
    theta = atan2(-v, w0 x)
    
    Then t ≈ (theta - phi0) / w0.
    The driving force is F0 cos(w_d t + phi_d).
    This makes the function highly non-linear and potentially multi-modal.
    
    To make BFGS robust, we might simplify the driving term estimation.
    Or, we can use a more flexible basis that doesn't rely on explicit time 
    reconstruction, but captures nonlinearity.
    
    Let's try a hybrid: 
    a = p0*x + p1*v + p2*x^3 + p3*x^2*v + p4*v^3 + p5*x*v^2 + p6*x^5 ...
    This is a polynomial basis. It's linear in params, which is great for BFGS.
    The previous attempt at polynomial basis (inspiration 3) had issues with 
    scaling or singular Hessian. 
    The finding says: "Removing redundant x^2 term fixes singular Hessian and 
    scaling inputs to O(1) improves BFGS convergence".
    
    Let's implement a carefully selected polynomial basis with scaling.
    Inputs x, v are in [-0.57, 0.53] and [-0.18, 0.28]. 
    We scale them to be O(1) to help BFGS.
    
    Basis:
    1. x
    2. v
    3. x^3 (nonlinear restoring)
    4. x^2 * v (damping nonlinearity)
    5. v^3 (velocity damping)
    6. x * v^2 (cross term)
    7. x^5 (higher order restoring)
    8. x^4 * v (higher order damping)
    9. v^5 (higher order damping)
    10. 1 (bias)
    
    We will scale x and v by constants to bring them to ~N(0,1) range 
    before applying the polynomial, or just use the raw values if they are 
    already small. 
    x ~ 0.5, v ~ 0.2. 
    x^3 ~ 0.125, x^5 ~ 0.03. 
    These are small. BFGS might struggle if params are large to compensate.
    Let's scale x by 2 and v by 5 to make them O(1).
    
    Let X = 2*x, V = 5*v.
    
    Basis terms on X, V:
    1. X
    2. V
    3. X^3
    4. X^2 * V
    5. V^3
    6. X * V^2
    7. X^5
    8. X^4 * V
    9. V^5
    10. 1
    
    This is linear in params. BFGS on linear least squares is convex.
    """
    # Scale inputs to O(1) to improve conditioning
    X = 2.0 * x
    V = 5.0 * v
    
    # Compute basis terms
    # Ensure finite inputs
    X = np.where(np.isfinite(X), X, 0.0)
    V = np.where(np.isfinite(V), V, 0.0)
    
    # Basis
    b1 = X
    b2 = V
    b3 = X**3
    b4 = X**2 * V
    b5 = V**3
    b6 = X * V**2
    b7 = X**5
    b8 = X**4 * V
    b9 = V**5
    b10 = 1.0
    
    # Combine with params
    # params[0] ... params[9]
    pred = (
        params[0] * b1 +
        params[1] * b2 +
        params[2] * b3 +
        params[3] * b4 +
        params[4] * b5 +
        params[5] * b6 +
        params[6] * b7 +
        params[7] * b8 +
        params[8] * b9 +
        params[9] * b10
    )
    
    return pred

# EVOLVE-BLOCK-END
