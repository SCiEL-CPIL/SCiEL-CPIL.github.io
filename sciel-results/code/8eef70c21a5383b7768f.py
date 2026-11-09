import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to center around 0 and scale to ~[-1, 1]
    # temp range [1, 40], center 20, scale 20 -> [-0.95, 1]
    # pH range [2, 12], center 7, scale 5 -> [-1, 1]
    # b range [1, 5], center 3, scale 2 -> [-1, 1]
    # s range [1, 5], use raw s for Monod but scale for consistency if needed? 
    # Monod works well with raw s.
    
    t = (temp - 20.0) / 20.0
    h = (pH - 7.0) / 5.0
    b_n = (b - 3.0) / 2.0
    
    # Ensure positive parameters for widths and scales
    # p1 -> K_s, p2 -> inv_sigma_t, p4 -> inv_sigma_h
    K_s = np.exp(p1)
    inv_sigma_t = np.exp(p2)
    inv_sigma_h = np.exp(p4)
    
    # Monod term
    monod = s / (K_s + s)
    
    # Temperature Gaussian term
    # p3 is the optimal temperature (normalized)
    temp_term = np.exp(-inv_sigma_t * (t - p3)**2)
    
    # pH Gaussian term
    # p5 is the optimal pH (normalized)
    ph_term = np.exp(-inv_sigma_h * (h - p5)**2)
    
    # Nutrient term (b)
    # Exponential of a quadratic to ensure positivity and flexibility
    # p6, p7, p8 are coefficients
    nutrient_term = np.exp(p6 + p7 * b_n + p8 * b_n**2)
    
    # Interaction term between temp and pH
    inter = 1.0 + p9 * t * h
    
    return p0 * monod * temp_term * ph_term * nutrient_term * inter
