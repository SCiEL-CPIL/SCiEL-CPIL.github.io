import numpy as np

def equation(b, s, temp, pH, params):
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure widths are positive for stability
    w_t = p3**2 + 1e-6
    w_ph = p5**2 + 1e-6
    
    # Basis functions
    # 1. Substrate: Monod-like saturation (fixed Ks=1 for simplicity, scaled by p1)
    # Using s/(1+s) is robust.
    f_s = s / (1.0 + s)
    
    # 2. Temperature: Gaussian centered at 37 (optimal for E. coli)
    # Amplitude p2, Width p3
    f_temp = p2 * np.exp(-((temp - 37.0)**2) / w_t)
    
    # 3. pH: Gaussian centered at 7.0
    # Amplitude p4, Width p5
    f_ph = p4 * np.exp(-((pH - 7.0)**2) / w_ph)
    
    # 4. Density: Exponential decay/inhibition
    # Rate p6. If p6 > 0, high density reduces growth.
    f_b = np.exp(-p6 * b)
    
    # 5. Interactions
    # Substrate-Temp: Does substrate availability modulate temperature sensitivity?
    f_st = p7 * f_s * (temp - 37.0)
    
    # Substrate-pH: Does substrate availability modulate pH sensitivity?
    f_sp = p8 * f_s * (pH - 7.0)
    
    # Temp-pH: Direct interaction between temperature and pH
    f_tp = p9 * (temp - 37.0) * (pH - 7.0)
    
    # Combine additively
    # p0 is the base bias
    y = p0 + p1 * f_s + f_temp + f_ph + f_b + f_st + f_sp + f_tp
    
    return y
