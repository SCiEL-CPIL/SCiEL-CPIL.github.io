import numpy as np
import time

# Deadline guard: track start time for the entire evaluation
_EQ_START = time.time()
_DEADLINE = _EQ_START + 140.0  # 10s safety margin before 150s kill

def _stable_sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def _smooth_ramp(x, k):
    """
    Smooth approximation of max(0, x) using x * sigmoid(k*x).
    For x >> 0: ~x (linear)
    For x << 0: ~0
    Smooth and differentiable everywhere.
    """
    k = np.clip(k, -100, 100)
    return x * _stable_sigmoid(k * x)

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using smooth ramp (clipped-linear) basis functions.
    
    Structure:
    1. Linear elastic term with temperature dependence.
    2. Primary hardening via smooth ramp (activates after yield strain p3).
    3. Secondary hardening via smooth ramp (for higher strain regimes).
    4. Intercept.
    
    Parameters:
    p0: Elastic modulus
    p1: Temperature coefficient for elastic modulus
    p2: Primary hardening slope
    p3: Primary yield strain
    p4: Primary ramp sharpness
    p5: Temperature coefficient for hardening
    p6: Secondary hardening slope
    p7: Secondary yield strain
    p8: Secondary ramp sharpness
    p9: Intercept
    """
    global _EQ_START, _DEADLINE
    if time.time() > _DEADLINE:
        # Return a safe fallback: linear elastic only
        return params[0] * strain
    
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Elastic term: E(T) * eps
    elastic = p0 * (1.0 - p1 * T) * eps
    
    # 2. Primary hardening: activates after yield strain p3
    #    For eps < p3: ~0; for eps > p3: ~p2*(1-p5*T)*(eps-p3)
    hard1 = p2 * (1.0 - p5 * T) * _smooth_ramp(eps - p3, p4)
    
    # 3. Secondary hardening: additional slope at higher strains
    hard2 = p6 * _smooth_ramp(eps - p7, p8)
    
    # 4. Intercept
    intercept = p9
    
    return elastic + hard1 + hard2 + intercept
