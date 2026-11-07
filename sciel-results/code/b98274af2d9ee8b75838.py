"""
Strategy: Localized RBF (Radial Basis Function) basis.
The current program uses a global polynomial series (powers of strain/temp).
Global polynomials are poor at fitting the sharp 'knee' (yield point) of 
metallic stress-strain curves because they tend to oscillate or require 
high degrees to capture local non-linearity.

This new strategy uses a combination of:
1. A linear elastic term (global stiffness).
2. A set of Gaussian "bumps" (RBFs) centered at specific strain values.
   These bumps allow the model to locally adjust the slope and offset 
   around the yield point and hardening region without affecting the 
   far-field elastic behavior.
3. Temperature-dependent scaling for the elastic term and bump amplitudes.

This acts as a flexible local basis function model, which is often more 
robust for fitting piecewise-linear or sharp-curve phenomena with a small 
number of parameters compared to global polynomials.
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a localized RBF basis.
    
    Parameters (10 total):
    p0: Elastic modulus (slope)
    p1: Temperature coefficient for elastic modulus
    p2: Offset (intercept)
    p3: Yield strain center 1 (approx location of knee)
    p4: Width of bump 1
    p5: Amplitude of bump 1
    p6: Yield strain center 2 (approx location of hardening)
    p7: Width of bump 2
    p8: Amplitude of bump 2
    p9: Temperature coefficient for bump amplitudes
    
    The model is:
    sigma = (p0 + p1*t) * s + p2 
            + (p5 + p9*t) * exp(-((s - p3)/p4)^2)
            + (p8 + p9*t) * exp(-((s - p6)/p7)^2)
            
    Note: We reuse p9 for simplicity in temp scaling of bumps, or we could 
    dedicate more params. Given 10 params, this structure balances 
    flexibility and identifiability.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # 1. Global Linear Elastic Term with Temperature Dependence
    # E(t) * s + offset
    elastic_term = (p0 + p1 * t) * s + p2
    
    # 2. Localized Non-linear Terms (Gaussian Bumps)
    # These capture the yield and hardening locally.
    # We need to handle division by zero or very small widths.
    # Clip widths to a small positive value to ensure numerical stability.
    w1 = np.clip(p4, 1e-4, None)
    w2 = np.clip(p7, 1e-4, None)
    
    # Bump 1: Around the yield point
    # Center at p3, width p4
    z1 = (s - p3) / w1
    # Clip z1 to avoid overflow in exp
    z1 = np.clip(z1, -10, 10)
    bump1 = np.exp(-z1 * z1)
    
    # Bump 2: Around the hardening region
    # Center at p6, width p7
    z2 = (s - p6) / w2
    z2 = np.clip(z2, -10, 10)
    bump2 = np.exp(-z2 * z2)
    
    # Amplitudes with temperature dependence
    # p5 is base amplitude for bump 1
    # p8 is base amplitude for bump 2
    # p9 scales the temperature effect on both amplitudes
    amp1 = p5 + p9 * t
    amp2 = p8 + p9 * t
    
    stress = elastic_term + amp1 * bump1 + amp2 * bump2
    
    return stress
