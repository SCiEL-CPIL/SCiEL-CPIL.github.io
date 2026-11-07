import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: 2-Gaussian Radial Basis Function (RBF) Network.
    
    The 10 parameters are interpreted as:
    - p[0:3]: Center of Gaussian 1 (cx1, cy1, cz1)
    - p[3:6]: Center of Gaussian 2 (cx2, cy2, cz2)
    - p[6]:   Width (inverse scale) of Gaussian 1 (sigma1)
    - p[7]:   Width (inverse scale) of Gaussian 2 (sigma2)
    - p[8]:   Weight/Amplitude of Gaussian 1 (w1)
    - p[9]:   Weight/Amplitude of Gaussian 2 (w2)
    
    Model: f(x,y,z) = w1 * exp(-||[x,y,z] - c1||^2 / (2*sigma1^2)) 
             + w2 * exp(-||[x,y,z] - c2||^2 / (2*sigma2^2))
    
    This local basis is highly flexible for capturing non-linear structures 
    within the bounded domain [0.5, 2]^3.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Unpack parameters
        # Centers
        cx1, cy1, cz1 = p[0], p[1], p[2]
        cx2, cy2, cz2 = p[3], p[4], p[5]
        
        # Widths (ensure positive to avoid division by zero or negative variance)
        # Use softplus-like transformation or clipping to keep sigma > 0
        # Since BFGS starts at 1, and we expect sigmas to be positive, 
        # we clip to a small positive value to maintain numerical stability.
        sigma1 = np.clip(p[6], 1e-3, 10.0)
        sigma2 = np.clip(p[7], 1e-3, 10.0)
        
        # Weights
        w1 = p[8]
        w2 = p[9]
        
        # Compute squared distances for Gaussian 1
        dx1 = x - cx1
        dy1 = y - cy1
        dz1 = z - cz1
        dist_sq1 = dx1*dx1 + dy1*dy1 + dz1*dz1
        
        # Compute Gaussian 1
        # Use np.exp with clipping on the exponent to prevent underflow/overflow issues
        # Exponent is -dist_sq / (2 * sigma^2)
        # Since dist_sq >= 0 and sigma^2 > 0, exponent is <= 0.
        # exp(0) = 1, exp(-large) -> 0.
        g1 = np.exp(-dist_sq1 / (2.0 * sigma1 * sigma1))
        
        # Compute squared distances for Gaussian 2
        dx2 = x - cx2
        dy2 = y - cy2
        dz2 = z - cz2
        dist_sq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
        
        # Compute Gaussian 2
        g2 = np.exp(-dist_sq2 / (2.0 * sigma2 * sigma2))
        
        # Combine
        output = w1 * g1 + w2 * g2
        
        # Safety check
        if not np.all(np.isfinite(output)):
            return np.zeros_like(x)
            
        return output
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
