import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    u = strain
    v = temp
    
    # Pure bivariate polynomial chaos expansion (Taylor series)
    # Using up to total degree 3 to capture non-linear interactions
    # while keeping the parameter space smooth and differentiable for BFGS.
    
    # Degree 0 (Intercept)
    t0 = p[0]
    
    # Degree 1
    t1_u = p[1] * u
    t1_v = p[2] * v
    
    # Degree 2
    t2_uu = p[3] * u**2
    t2_uv = p[4] * u * v
    t2_vv = p[5] * v**2
    
    # Degree 3
    t3_uuu = p[6] * u**3
    t3_uuv = p[7] * u**2 * v
    t3_uvv = p[8] * u * v**2
    t3_vvv = p[9] * v**3
    
    stress = t0 + t1_u + t1_v + t2_uu + t2_uv + t2_vv + t3_uuu + t3_uuv + t3_uvv + t3_vvv
    
    return stress
