import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)

    abs_strain = np.abs(strain)
    # Smooth sign to ensure differentiability
    sign_strain = np.tanh(strain / 1e-4)
    
    T = temp
    
    # Coefficients
    c1 = p[0] + p[1] * T
    c2 = p[2] + p[3] * T
    c3 = p[4] + p[5] * T
    c4 = p[6] + p[7] * T
    c5 = p[8]
    c6 = p[9]
    
    # Polynomial in absolute strain
    s2 = abs_strain * abs_strain
    s3 = s2 * abs_strain
    s4 = s2 * s2
    s5 = s4 * abs_strain
    s6 = s3 * s3
    
    poly = c1 * abs_strain + c2 * s2 + c3 * s3 + c4 * s4 + c5 * s5 + c6 * s6
    
    sigma = sign_strain * poly
    
    return np.where(np.isfinite(sigma), sigma, 0.0)
