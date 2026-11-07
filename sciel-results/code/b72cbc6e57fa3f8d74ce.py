import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model for E. coli growth rate.
    
    Model Structure:
    y = p0 * (s / (exp(p1) + s)) * 
        exp(-((temp - (p2 + p8*(pH - p4)))^2) / (exp(p3)^2)) * 
        exp(-((pH - (p4 + p9*(temp - p2)))^2) / (exp(p5)^2)) * 
        (1 + p6*b + p7*b^2)
        
    Parameters:
    p0: Global scale factor
    p1: Log of substrate half-saturation constant (K_s)
    p2: Base optimal temperature
    p3: Log of temperature width (sigma_T)
    p4: Base optimal pH
    p5: Log of pH width (sigma_pH)
    p6: Linear density coefficient
    p7: Quadratic density coefficient
    p8: Shift in optimal temperature due to pH deviation
    p9: Shift in optimal pH due to temperature deviation
    """
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # 1. Substrate Saturation (Monod)
    # Ensure K_s is positive and stable
    K_s = np.exp(np.clip(p1, -20.0, 20.0))
    u_s = s / (K_s + s)
    
    # 2. Temperature Response (Gaussian)
    # Optimal temperature shifts with pH
    T_opt = p2 + p8 * (pH - p4)
    # Width is positive
    sigma_T = np.exp(np.clip(p3, -10.0, 10.0))
    # Prevent division by zero
    sigma_T_safe = np.maximum(sigma_T, 1e-6)
    
    # Calculate exponent, clip to avoid underflow/overflow issues in exp
    exp_T = -((temp - T_opt) ** 2) / (sigma_T_safe ** 2)
    exp_T = np.clip(exp_T, -500.0, 0.0)
    y_T = np.exp(exp_T)
    
    # 3. pH Response (Gaussian)
    # Optimal pH shifts with Temperature
    pH_opt = p4 + p9 * (temp - p2)
    # Width is positive
    sigma_pH = np.exp(np.clip(p5, -10.0, 10.0))
    sigma_pH_safe = np.maximum(sigma_pH, 1e-6)
    
    exp_pH = -((pH - pH_opt) ** 2) / (sigma_pH_safe ** 2)
    exp_pH = np.clip(exp_pH, -500.0, 0.0)
    y_pH = np.exp(exp_pH)
    
    # 4. Density Effect
    # Quadratic term allows for both promotion and inhibition
    y_b = 1.0 + p6 * b + p7 * b ** 2
    
    # Combine all factors
    result = p0 * u_s * y_T * y_pH * y_b
    
    # Ensure output is finite
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
