"""Kinked piecewise-linear model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = E_eff * strain + H1_eff * max(0, strain - y1) + H2_eff * max(0, strain - y2)
    
    This model uses a ReLU (Rectified Linear Unit) basis to create kinks in the stress-strain curve.
    - The first term is the linear elastic response.
    - The second term activates at strain y1 (yield point), adding a slope change (plastic flow).
    - The third term activates at strain y2 (hardening knee), adding a further slope change (work hardening).
    
    Parameters:
    p0, p1: E_eff = p0 + p1*temp (Elastic modulus, temperature-dependent)
    p2, p3: H1_eff = p2 + p3*temp (Plastic flow slope, temperature-dependent)
    p4, p5: H2_eff = p4 + p5*temp (Work hardening slope, temperature-dependent)
    p6, p7: y1 = exp(p6) + p7*temp (Yield strain, log-parameterized for positivity, temp shift)
            Note: Using exp(p6) ensures y1 > 0. p7 allows temp to shift the yield point.
            To keep y1 in a reasonable range for BFGS starting at 1:
            exp(1) ~ 2.7 is too high. We want y1 ~ 0.002-0.01.
            Let's use y1 = exp(p6) * 0.01 + p7*temp. 
            If p6=1, exp(1)*0.01 ~ 0.027. Still a bit high.
            Let's use y1 = exp(p6) * 0.001 + p7*temp.
            If p6=1, 0.0027. This is good.
            Similarly for y2.
    p8, p9: y2 = exp(p8) * 0.01 + p9*temp (Hardening knee strain)
            If p8=1, 0.027.
            
    Let's refine the parameterization to be safe for BFGS starting at 1.
    p6, p8 are exponents. 
    y1 = 0.001 * np.exp(p6) + p7 * temp
    y2 = 0.01 * np.exp(p8) + p9 * temp
    
    If p6=1, y1 ~ 0.0027 + 1*temp. Temp is ~1. So y1 ~ 1.0027? No, temp is [0.06, 1].
    Wait, temp is normalized? "Inputs in training range... [0.06667, 1]". 
    So temp is roughly 1.
    If p7=1, y1 = 0.0027 + 1*1 = 1.0027. This is outside the strain range [0, 0.96] or close to the end.
    We want y1 to be small, e.g., 0.005.
    So p7 should be small or negative. BFGS will figure it out.
    But starting at 1 might be bad if the initial guess is far.
    
    Alternative: 
    y1 = 0.01 / (1 + np.exp(-p6)) + p7*temp  # Sigmoid centered at 0.01
    If p6=1, sigmoid(1) ~ 0.73. y1 ~ 0.0073 + p7.
    If p7=1, y1 ~ 1.007. Still bad.
    
    Let's just use direct linear parameters for the kinks but constrain them to be positive via softplus or exp, but scale them appropriately.
    
    y1 = np.exp(p6) * 0.005 + p7 * temp
    y2 = np.exp(p8) * 0.05 + p9 * temp
    
    If p6=1, exp(1)*0.005 ~ 0.0135.
    If p7=1, y1 ~ 0.0135 + 1. This is bad.
    
    The issue is that BFGS starts at 1. If a parameter represents a small offset, starting at 1 is huge.
    However, the task says "starting from all ones".
    We must design the function so that params=1 gives a reasonable initial guess.
    
    Let's try:
    y1 = 0.002 * np.exp(p6) + 0.001 * p7 * temp
    y2 = 0.02 * np.exp(p8) + 0.01 * p9 * temp
    
    If p6=1, p7=1, temp=1: y1 = 0.002*2.7 + 0.001 = 0.0065. Good.
    If p8=1, p9=1, temp=1: y2 = 0.02*2.7 + 0.01 = 0.064. Good.
    
    Slopes:
    E = p0 + p1*temp. If p0=1, p1=1, temp=1, E=2. Reasonable for normalized stress?
    Target range is [-0.005, 0.91]. Strain up to 0.96.
    If E=2, stress at strain 0.006 is 0.012.
    Plastic slope H1 = p2 + p3*temp. If p2=1, p3=1, H1=2.
    Hardening H2 = p4 + p5*temp. If p4=1, p5=1, H2=2.
    
    This seems like a reasonable starting point.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # Elastic Modulus
    E = p0 + p1 * temp
    # Plastic Flow Slope (change in slope after yield)
    H1 = p2 + p3 * temp
    # Work Hardening Slope (change in slope after knee)
    H2 = p4 + p5 * temp
    
    # Yield Strain (Kink 1)
    # Scaled to be around 0.005-0.02 for initial params
    y1 = 0.002 * np.exp(p6) + 0.001 * p7 * temp
    
    # Hardening Knee Strain (Kink 2)
    # Scaled to be around 0.02-0.1 for initial params
    y2 = 0.02 * np.exp(p8) + 0.01 * p9 * temp
    
    # ReLU basis functions
    # max(0, strain - y)
    k1 = np.maximum(0.0, strain - y1)
    k2 = np.maximum(0.0, strain - y2)
    
    # Stress = E*strain + H1*k1 + H2*k2
    stress = E * strain + H1 * k1 + H2 * k2
    
    return stress
# EVOLVE-BLOCK-END
