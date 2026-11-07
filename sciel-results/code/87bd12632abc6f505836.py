import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF basis for nonlinear oscillator acceleration.
    Uses 9 radial basis functions centered at diverse points in (x,v) space,
    plus a linear term for global trend. This structure captures local nonlinearities
    better than polynomial bases.
    """
    # Normalize inputs to standard scale for stable RBF computation
    X = x * 1.75  # roughly standardize: range [-1.0, 0.92]
    V = v * 3.33  # roughly standardize: range [-0.58, 0.94]
    
    # Precompute squared distances to each center for efficiency
    # Centers chosen to cover the input domain with some overlap
    
    # Center 1: origin
    d1 = X * X + V * V
    
    # Center 2: positive x, positive v
    cx2, cv2 = 0.5, 0.5
    d2 = (X - cx2) ** 2 + (V - cv2) ** 2
    
    # Center 3: negative x, positive v
    cx3, cv3 = -0.5, 0.5
    d3 = (X - cx3) ** 2 + (V - cv3) ** 2
    
    # Center 4: positive x, negative v
    cx4, cv4 = 0.5, -0.5
    d4 = (X - cx4) ** 2 + (V - cv4) ** 2
    
    # Center 5: negative x, negative v
    cx5, cv5 = -0.5, -0.5
    d5 = (X - cx5) ** 2 + (V - cv5) ** 2
    
    # Center 6: positive x, zero v
    cx6, cv6 = 0.6, 0.0
    d6 = (X - cx6) ** 2 + (V - cv6) ** 2
    
    # Center 7: negative x, zero v
    cx7, cv7 = -0.6, 0.0
    d7 = (X - cx7) ** 2 + (V - cv7) ** 2
    
    # Center 8: zero x, positive v
    cx8, cv8 = 0.0, 0.6
    d8 = (X - cx8) ** 2 + (V - cv8) ** 2
    
    # Center 9: zero x, negative v
    cx9, cv9 = 0.0, -0.6
    d9 = (X - cx9) ** 2 + (V - cv9) ** 2
    
    # Gaussian RBFs with width parameter 0.8
    width_sq = 2.0 * 0.8 ** 2  # 2 * sigma^2
    r1 = np.exp(-d1 / width_sq)
    r2 = np.exp(-d2 / width_sq)
    r3 = np.exp(-d3 / width_sq)
    r4 = np.exp(-d4 / width_sq)
    r5 = np.exp(-d5 / width_sq)
    r6 = np.exp(-d6 / width_sq)
    r7 = np.exp(-d7 / width_sq)
    r8 = np.exp(-d8 / width_sq)
    r9 = np.exp(-d9 / width_sq)
    
    # Linear terms for global trend
    lin_x = X
    lin_v = V
    
    # Build feature matrix
    T = np.empty((x.shape[0], 10), dtype=np.float64)
    T[:, 0] = r1
    T[:, 1] = r2
    T[:, 2] = r3
    T[:, 3] = r4
    T[:, 4] = r5
    T[:, 5] = r6
    T[:, 6] = r7
    T[:, 7] = r8
    T[:, 8] = r9
    T[:, 9] = lin_x * 0.5 + lin_v * 0.5  # combine linear terms into one param
    
    return T @ params
