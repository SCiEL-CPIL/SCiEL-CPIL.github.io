import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Model Structure:
    y = Base * Substrate_Kinetics * Temp_Response * pH_Response * Density_Inhibition + Offset
    
    Parameters (mapped from params[0..9] starting at 1.0):
    p0: Base rate scale
    p1: Ks (half saturation)
    p2: Temp Center
    p3: Temp Variance
    p4: pH Center
    p5: pH Variance
    p6: Density Inhibition Rate
    p7: Substrate Inhibition Constant
    p8: Temp Skew (linear correction)
    p9: Offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Mapping parameters to physical ranges
    # Initial guess params are all 1.0
    
    # Base rate: allow negative? No, growth rate can be negative (net decline). 
    # But the multiplicative part should be positive.
    base_rate = p0
    
    # Substrate: Monod with inhibition
    # Ks = 1.0 when p1=1. Range maybe 0.1 to 100.
    Ks = np.exp(p1 - 1.0) # If p1=1, Ks=1. If p1=0, Ks=0.36. If p1=2, Ks=2.7.
    
    # Substrate Inhibition: s^2 / (Ks * p7_mapped)
    # Let p7_mapped be the inhibition constant.
    # If p7=1, let's say inhibition starts at high s.
    s_inh_const = np.exp(p7 - 1.0) * 10.0 # Scale up to make it relevant in range [1, 40]
    
    substrate_term = s / (Ks + s + (s * s) / s_inh_const)
    
    # Temperature: Gaussian with Skew
    # Center: 20 + 10*(p2-1). If p2=1, C=20.
    T_center = 20.0 + 10.0 * (p2 - 1.0)
    # Variance: 10 + 5*(p3-1). If p3=1, V=10.
    T_var = 10.0 + 5.0 * (p3 - 1.0)
    # Ensure T_var is positive
    T_var = np.maximum(T_var, 1e-6)
    
    # Skew: 0.1*(p8-1). If p8=1, Skew=0.
    T_skew = 0.1 * (p8 - 1.0)
    
    # Temp term: exp(- (T - C)^2 / (2*V)) * (1 + Skew*(T - C))
    # The linear term can make it negative if T is far from C. 
    # To keep it stable, we might clip or use a different skew.
    # However, for BFGS, non-smoothness from clipping is bad.
    # Let's assume the skew is small enough or the Gaussian decays fast enough.
    # Or use a shifted Gaussian: exp(- ((T - C) - Skew*(T-C)^2/V)^2 / (2*V) )? Too complex.
    # Let's stick to the linear multiplier but ensure it doesn't cause huge negatives.
    # Actually, if the model allows negative growth, the linear term is fine.
    
    temp_term = np.exp(-((temp - T_center) ** 2) / (2.0 * T_var)) * (1.0 + T_skew * (temp - T_center))
    
    # pH: Gaussian
    # Center: 7 + 2*(p4-1). If p4=1, C=7.
    pH_center = 7.0 + 2.0 * (p4 - 1.0)
    # Variance: 2 + 1*(p5-1). If p5=1, V=2.
    pH_var = 2.0 + 1.0 * (p5 - 1.0)
    pH_var = np.maximum(pH_var, 1e-6)
    
    pH_term = np.exp(-((pH - pH_center) ** 2) / (2.0 * pH_var))
    
    # Density: Exponential Inhibition
    # Rate: 0.1*(p6-1). If p6=1, Rate=0.
    # If rate is 0, no inhibition. If rate > 0, inhibition.
    # If rate < 0, enhancement (unlikely).
    b_rate = 0.1 * (p6 - 1.0)
    density_term = np.exp(-b_rate * b)
    
    # Offset
    offset = 0.1 * (p9 - 1.0)
    
    # Combine
    y = base_rate * substrate_term * temp_term * pH_term * density_term + offset
    
    return y
