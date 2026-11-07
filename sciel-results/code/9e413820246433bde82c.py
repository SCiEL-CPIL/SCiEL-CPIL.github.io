import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Params
    E0 = params[0]
    E1 = params[1]
    E2 = params[9]
    offset = params[2]
    
    w0 = params[3]
    w1 = params[4]
    w2 = params[5]
    
    v0 = params[6]
    v1 = params[7]
    v2 = params[8]
    
    # Fixed centers and widths
    c0 = 0.002
    c1 = 0.05
    c2 = 0.5
    k = 100.0
    
    # Elastic part
    E_T = E0 + E1 * temp + E2 * temp * temp
    sigma_elastic = E_T * strain
    
    # Plastic part (RBFs)
    rbf0 = np.exp(-k * (strain - c0)**2)
    rbf1 = np.exp(-k * (strain - c1)**2)
    rbf2 = np.exp(-k * (strain - c2)**2)
    
    sigma_plastic = (w0 + v0 * temp) * rbf0 + (w1 + v1 * temp) * rbf1 + (w2 + v2 * temp) * rbf2
    
    return sigma_elastic + sigma_plastic + offset
