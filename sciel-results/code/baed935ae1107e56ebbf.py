import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to a centered/scaled space for better conditioning
    # Training ranges: b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12]
    # Centers: b_c=3, s_c=3, T_c=20, pH_c=7
    # Scales: b_s=2, s_s=2, T_s=20, pH_s=5
    
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    T_n = (temp - 20.0) / 20.0
    pH_n = (pH - 7.0) / 5.0
    
    # Monod saturation in original space, but we can use normalized s for flexibility
    # Use a general saturation: u_s = s / (K_s + s) where K_s is fitted
    # To keep it within 10 params and stable, let's use:
    # u_s = s / (1.0 + params[0] * s)  -> but this might be hard to fit if s is large
    # Better: u_s = s / (K_s + s)
    # Let K_s = params[0] + 0.1 to ensure positive K_s
    
    K_s = params[0] + 0.1
    u_s = s / (K_s + s)
    
    # Temperature polynomial (quadratic) in normalized space
    # T_n is centered, so linear term captures slope, quadratic captures curvature
    t_poly = params[1] + params[2] * T_n + params[3] * T_n**2
    
    # pH polynomial (quadratic) in normalized space
    pH_poly = params[4] + params[5] * pH_n + params[6] * pH_n**2
    
    # Density effect (linear) in normalized space
    y_b = params[7] * b_n
    
    # Interactions: 
    # Substrate-gated temperature effect
    y_s_T = params[8] * u_s * T_n
    
    # Substrate-gated pH effect
    y_s_pH = params[9] * u_s * pH_n
    
    # Combine: 
    # Main growth is substrate-gated temperature effect
    # Plus pH effect (can be substrate independent or gated? Let's keep it additive for now)
    # Plus density
    # Plus interactions
    
    y = u_s * t_poly + pH_poly + y_b + y_s_T + y_s_pH
    
    return y
