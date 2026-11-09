import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Scale x to [-1, 1] for Chebyshev conditioning
    t = x / 2.0
    
    # Chebyshev polynomials of the first kind T_n(t)
    # T_0 = 1, T_1 = t, T_2 = 2t^2 - 1, T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1, T_5 = 16t^5 - 20t^3 + 5t
    # T_6 = 32t^6 - 48t^4 + 18t^2 - 1
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    T6 = 32.0 * t**6 - 48.0 * t**4 + 18.0 * t**2 - 1.0
    
    # 3 Gaussian RBF terms at strategic locations in original x-space
    rbf_centers = np.array([-1.0, 0.0, 1.0])
    rbf_widths = 0.7
    
    rbf0 = np.exp(-((x - rbf_centers[0])**2) / (2.0 * rbf_widths**2))
    rbf1 = np.exp(-((x - rbf_centers[1])**2) / (2.0 * rbf_widths**2))
    rbf2 = np.exp(-((x - rbf_centers[2])**2) / (2.0 * rbf_widths**2))
    
    # Combine: 7 Chebyshev + 3 RBF = 10 terms
    y_hat = (p[0] * T0 + 
             p[1] * T1 + 
             p[2] * T2 + 
             p[3] * T3 + 
             p[4] * T4 + 
             p[5] * T5 + 
             p[6] * T6 + 
             p[7] * rbf0 + 
             p[8] * rbf1 + 
             p[9] * rbf2)
             
    return y_hat
