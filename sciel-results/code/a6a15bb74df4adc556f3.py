import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Anisotropic RBF + polynomial model for nonlinear oscillator acceleration.
    
    Structure: a = f(x, v) = polynomial_basis + sum of Anisotropic Gaussian RBFs
    
    The polynomial part captures global trends (restoring, damping).
    The RBF part provides localized corrections with anisotropic shapes,
    better capturing the elliptical nature of limit cycles in phase space.
    
    Parameters (10):
      p0: constant bias
      p1: x coefficient (linear restoring)
      p2: v coefficient (linear damping)
      p3: x^2 coefficient
      p4: v^2 coefficient
      p5: RBF amplitude 1
      p6: RBF amplitude 2
      p7: RBF amplitude 3
      p8: RBF width in x (shared sigma_x)
      p9: RBF width in v (shared sigma_v)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis: low-order terms for global structure
    poly = p0 + p1 * x + p2 * v + p3 * x * x + p4 * v * v
    
    # RBF centers (fixed, not fitted)
    # Chosen to cover the input domain asymmetrically
    cx1, cv1 = 0.0, 0.0
    cx2, cv2 = 0.3, 0.15
    cx3, cv3 = -0.3, -0.1
    
    # Compute squared distances to each center
    dx1 = x - cx1
    dv1 = v - cv1
    
    dx2 = x - cx2
    dv2 = v - cv2
    
    dx3 = x - cx3
    dv3 = v - cv3
    
    # Shared anisotropic width parameters p8 (sigma_x) and p9 (sigma_v)
    # Use abs to keep sigmas positive
    sigma_x = np.abs(p8) + 1e-6
    sigma_v = np.abs(p9) + 1e-6
    
    sigma_x2 = 2.0 * sigma_x * sigma_x
    sigma_v2 = 2.0 * sigma_v * sigma_v
    
    # Anisotropic Gaussian RBFs
    rbf1 = np.exp(-(dx1 * dx1 / sigma_x2) - (dv1 * dv1 / sigma_v2))
    rbf2 = np.exp(-(dx2 * dx2 / sigma_x2) - (dv2 * dv2 / sigma_v2))
    rbf3 = np.exp(-(dx3 * dx3 / sigma_x2) - (dv3 * dv3 / sigma_v2))
    
    result = poly + p5 * rbf1 + p6 * rbf2 + p7 * rbf3
    
    return result
