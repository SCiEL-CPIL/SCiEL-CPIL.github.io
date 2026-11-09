import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Ensure params are finite to avoid NaNs in the model
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # 10 parameters:
    # p[0]: global offset (bias)
    # p[1]: amplitude for x-bump
    # p[2]: center for x-bump
    # p[3]: width for x-bump
    # p[4]: amplitude for y-bump
    # p[5]: center for y-bump
    # p[6]: width for y-bump
    # p[7]: amplitude for z-bump
    # p[8]: center for z-bump
    # p[9]: width for z-bump

    # Calculate widths, ensuring they are positive and not too small to avoid division issues
    # Adding a small epsilon to prevent zero division or extreme peaks
    wx = p[3] + 1e-3
    wy = p[6] + 1e-3
    wz = p[9] + 1e-3
    
    # Calculate RBF terms
    # Using exp(-(x-c)^2 / w^2)
    term_x = p[1] * np.exp(-((x - p[2]) / wx) ** 2)
    term_y = p[4] * np.exp(-((y - p[5]) / wy) ** 2)
    term_z = p[7] * np.exp(-((z - p[8]) / wz) ** 2)
    
    # Combine with global offset
    result = p[0] + term_x + term_y + term_z
    
    # Ensure result is finite
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
