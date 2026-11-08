import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [-1, 1] based on known training ranges
    # b: [1, 4.998] -> center 2.999, half-range ~2
    # s: [1, 4.999] -> center 2.9995, half-range ~2
    # temp: [1.002, 39.98] -> center 20.49, half-range ~19.5
    # pH: [2, 12] -> center 7, half-range 5
    
    b_n = (b - 2.999) / 2.0
    s_n = (s - 2.9995) / 2.0
    t_n = (temp - 20.491) / 19.49
    p_n = (pH - 7.0) / 5.0
    
    # Clip to avoid extreme values if slightly out of range, though inputs are in range
    # b_n = np.clip(b_n, -1.5, 1.5)
    # s_n = np.clip(s_n, -1.5, 1.5)
    # t_n = np.clip(t_n, -1.5, 1.5)
    # p_n = np.clip(p_n, -1.5, 1.5)
    
    # Construct 10 polynomial basis functions
    # We use 9 features and 1 bias term inside the linear combination
    # Features:
    # 1: b_n
    # 2: s_n
    # 3: t_n
    # 4: p_n
    # 5: b_n^2
    # 6: t_n^2
    # 7: p_n^2
    # 8: b_n * t_n
    # 9: b_n * p_n
    # Bias is added separately or as 10th param multiplying 1
    
    phi_1 = b_n
    phi_2 = s_n
    phi_3 = t_n
    phi_4 = p_n
    phi_5 = b_n * b_n
    phi_6 = t_n * t_n
    phi_7 = p_n * p_n
    phi_8 = b_n * t_n
    phi_9 = b_n * p_n
    
    # Linear combination with 10 parameters
    # p[0] to p[8] for features, p[9] for bias
    z = (p[0] * phi_1 + 
         p[1] * phi_2 + 
         p[2] * phi_3 + 
         p[3] * phi_4 + 
         p[4] * phi_5 + 
         p[5] * phi_6 + 
         p[6] * phi_7 + 
         p[7] * phi_8 + 
         p[8] * phi_9 + 
         p[9])
         
    # Softplus activation to ensure non-negative growth rate and smoothness
    # np.log1p(np.exp(z)) is stable for large z
    # For very negative z, it behaves like exp(z) ~ 0
    # For large positive z, it behaves like z
    return np.log1p(np.exp(z))
