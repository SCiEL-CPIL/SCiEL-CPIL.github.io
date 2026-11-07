# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using 10 localized Gaussian basis functions
    centered at evenly spaced points across [-3, 3].
    
    params[0..9] are coefficients for:
    exp(-((x - c_i) / sigma)^2) for i = 0..9
    where c_i = -3 + 6*i/9 and sigma is a fixed width.
    
    This provides local flexibility and avoids ill-conditioning
    of global polynomial bases.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 10 centers evenly spaced from -3 to 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width for Gaussian basis functions
    # Width should be large enough to have overlap but small enough for local flexibility
    # With 10 centers over range 6, spacing is 0.667
    # A width of about 0.8 gives good overlap between neighbors
    sigma = 0.8
    
    # Compute Gaussian basis functions: exp(-((x - c_i) / sigma)^2)
    # Shape: (n_points, 10)
    basis = np.exp(-((x[:, np.newaxis] - centers[np.newaxis, :]) / sigma) ** 2)
    
    # Linear combination
    y = basis @ np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
    
    return y
# EVOLVE-BLOCK-END
