import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 params:
    # p0, p1: center 1 (x, y)
    # p2: width 1 (mapped to positive)
    # p3: amplitude 1
    # p4, p5: center 2 (x, y)
    # p6: width 2 (mapped to positive)
    # p7: amplitude 2
    # p8, p9: Let's add a 3rd RBF with fixed center (0,0) and optimizable width/amplitude?
    # p8: width 3
    # p9: amplitude 3
    
    # Actually, let's use 2 full RBFs (8 params) and 2 params for a linear trend?
    # p8: slope_x
    # p9: slope_y
    # This captures global linear trends + local bumps.
    
    cx1, cy1 = params[0], params[1]
    sigma1_raw = params[2]
    a1 = params[3]
    
    cx2, cy2 = params[4], params[5]
    sigma2_raw = params[6]
    a2 = params[7]
    
    slope_x = params[8]
    slope_y = params[9]
    
    # Map widths to positive values
    # Using exp ensures positivity. To avoid overflow, we can scale.
    # If param is 1, exp(1) ~ 2.7. If param is -5, exp(-5) ~ 0.006.
    # Let's use: sigma = 0.1 + 0.9 * sigmoid(param)
    # This keeps sigma in [0.1, 1.0].
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    
    sigma1 = 0.1 + 0.9 * sigmoid(sigma1_raw)
    sigma2 = 0.1 + 0.9 * sigmoid(sigma2_raw)
    
    # Calculate RBFs
    # r1^2 = (x-cx1)^2 + (y-cy1)^2
    r1_sq = (x - cx1)**2 + (y - cy1)**2
    rbf1 = a1 * np.exp(-r1_sq / (2 * sigma1**2))
    
    r2_sq = (x - cx2)**2 + (y - cy2)**2
    rbf2 = a2 * np.exp(-r2_sq / (2 * sigma2**2))
    
    # Linear part
    linear = slope_x * x + slope_y * y
    
    return rbf1 + rbf2 + linear
