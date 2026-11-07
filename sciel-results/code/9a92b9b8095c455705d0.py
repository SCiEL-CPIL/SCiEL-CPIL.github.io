import numpy as np
import time

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Physics-Informed Elastic-Plastic Model with Smooth Yield.
    
    This model explicitly separates the elastic and plastic regimes:
    1. Elastic: Hooke's Law, sigma = E * strain.
    2. Plastic: Hardening after yield, sigma = E * strain + H * (strain - s_y)^p.
    
    To maintain differentiability for BFGS, we use a smooth approximation for the 
    "plastic strain" term: (x - y)_+ is approximated by softplus(k * (x - y)) / k.
    Actually, a better smooth approximation for the *yield behavior* is to use a 
    sigmoid switch or a softplus directly on the strain excess.
    
    Let's use:
    sigma = E(t) * s + H(t) * softplus(k * (s - s_y(t)))^p
    
    However, softplus(x) = log(1 + exp(x)). For large negative x, it is ~0. 
    For large positive x, it is ~x.
    So softplus(k * (s - s_y)) approximates max(0, k * (s - s_y)).
    Dividing by k gives max(0, s - s_y).
    So: sigma = E(t) * s + H(t) * (softplus(k * (s - s_y(t))) / k)^p
    
    Parameters (10 total):
    p[0]: E_base (Young's modulus base)
    p[1]: E_temp (Young's modulus temp coeff)
    p[2]: H_base (Hardening modulus base)
    p[3]: H_temp (Hardening modulus temp coeff)
    p[4]: s_y_base (Yield strain base)
    p[5]: s_y_temp (Yield strain temp coeff)
    p[6]: k (Sharpness of yield transition)
    p[7]: p (Hardening exponent)
    p[8]: offset_base (Stress offset base)
    p[9]: offset_temp (Stress offset temp coeff)
    
    Note: BFGS starts at all ones. 
    E ~ 1, H ~ 1, s_y ~ 1, k ~ 1, p ~ 1.
    This might be far from optimal (E for Al is ~70 GPa, but strain is normalized? 
    The input strain is in [-0.0019, 0.96]. Stress is [-0.0055, 0.91].
    So E ~ Stress/Strain ~ 0.9/0.9 ~ 1. So scale is okay.)
    """
    s = strain
    t = temp
    p = params
    
    # Unpack parameters
    E = p[0] + p[1] * t
    H = p[2] + p[3] * t
    s_y = p[4] + p[5] * t
    k = p[6]
    power = p[7]
    offset = p[8] + p[9] * t
    
    # Calculate plastic strain term
    # softplus(x) = log1p(exp(x))
    # To avoid overflow in exp, we can use np.where or a stable log1p.
    # For x > 30, exp(x) is huge. log1p(exp(x)) ~ x.
    # For x < -30, exp(x) is tiny. log1p(exp(x)) ~ exp(x).
    
    x = k * (s - s_y)
    
    # Stable softplus
    # softplus(x) = max(x, 0) + log1p(exp(-abs(x)))
    # This is stable.
    softplus_val = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    
    # Plastic strain = softplus_val / k
    # If k is very small, this might be unstable, but BFGS should find reasonable k.
    # If k is negative? Softplus usually assumes k>0 for sharpness. 
    # Let's force k to be positive by using abs(k) or exp(k). 
    # Using exp(k) ensures k>0 and differentiability.
    k_eff = np.exp(p[6])
    
    # Recalculate with k_eff
    x = k_eff * (s - s_y)
    softplus_val = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    
    plastic_strain = softplus_val / k_eff
    
    # Hardening term
    # power should be positive. Use exp(p[7]) to ensure p>0.
    p_eff = np.exp(p[7])
    
    # Handle power of zero or negative if plastic_strain is 0.
    # np.power(0, p) is 0 for p>0.
    hardening = H * np.power(plastic_strain, p_eff)
    
    # Total stress
    stress = E * s + hardening + offset
    
    return stress
