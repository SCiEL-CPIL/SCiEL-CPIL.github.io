"""Kinked piecewise-linear model with smooth ReLU surrogate for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = E_eff * strain + H1_eff * soft_relu(strain - y1) + H2_eff * soft_relu(strain - y2)
    
    This model uses a smooth ReLU (softplus) basis to create kinks in the stress-strain curve.
    The softplus function is C-infinity smooth, which helps BFGS convergence by providing
    well-defined gradients everywhere (unlike hard ReLU which has a kink).
    
    - The first term is the linear elastic response.
    - The second term activates smoothly around strain y1 (yield point), adding a slope change.
    - The third term activates smoothly around strain y2 (hardening knee), adding a further slope change.
    
    Parameters:
    p0, p1: E_eff = p0 + p1*temp (Elastic modulus, temperature-dependent)
    p2, p3: H1_eff = p2 + p3*temp (Plastic flow slope, temperature-dependent)
    p4, p5: H2_eff = p4 + p5*temp (Work hardening slope, temperature-dependent)
    p6, p7: y1 = 0.002 * exp(p6) + 0.001 * p7 * temp (Yield strain)
    p8, p9: y2 = 0.02 * exp(p8) + 0.01 * p9 * temp (Hardening knee strain)
    
    For params=1 (BFGS start):
    y1 = 0.002*2.718 + 0.001*1 ≈ 0.0064
    y2 = 0.02*2.718 + 0.01*1 ≈ 0.0644
    These are reasonable starting points for aluminium yield/hardening strains.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # Elastic Modulus
    E = p0 + p1 * temp
    # Plastic Flow Slope (change in slope after yield)
    H1 = p2 + p3 * temp
    # Work Hardening Slope (change in slope after knee)
    H2 = p4 + p5 * temp
    
    # Yield Strain (Kink 1) - scaled for reasonable initial values
    y1 = 0.002 * np.exp(p6) + 0.001 * p7 * temp
    
    # Hardening Knee Strain (Kink 2) - scaled for reasonable initial values
    y2 = 0.02 * np.exp(p8) + 0.01 * p9 * temp
    
    # Smooth ReLU (softplus) basis functions
    # softplus(x) = log(1 + exp(x)), which approximates max(0, x) for large |x|
    # but is smooth everywhere, aiding BFGS optimization
    # Use a temperature-independent smoothing scale for numerical stability
    # The smoothing scale should be small relative to the strain range to approximate ReLU
    # but large enough to maintain smoothness. Strain range is ~1.0, so 0.01 is reasonable.
    # We can make it slightly adaptive or fixed. Let's use a fixed small scale.
    scale = 0.01
    
    # Compute softplus for (strain - y1) and (strain - y2)
    # To avoid overflow in exp for large arguments, use the stable log1p(exp(x)) formulation
    # or clip the argument. Since strain is in [-0.002, 0.96] and y1, y2 are small positive,
    # (strain - y) is in a reasonable range. But during optimization, params might go wild.
    # Let's clip the argument to [-50, 50] to prevent overflow in exp.
    
    arg1 = (strain - y1) / scale
    arg2 = (strain - y2) / scale
    
    # Stable softplus: log(1 + exp(x))
    # For x > 0: x + log1p(exp(-x))
    # For x <= 0: log1p(exp(x))
    # We can use np.logaddexp(0, x) which is stable.
    
    k1 = scale * np.logaddexp(0.0, arg1)
    k2 = scale * np.logaddexp(0.0, arg2)
    
    # Stress = E*strain + H1*k1 + H2*k2
    stress = E * strain + H1 * k1 + H2 * k2
    
    return stress
# EVOLVE-BLOCK-END
