import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a learnable Radial Basis Function (RBF) network.
    
    The 10 parameters are split into:
    - 6 parameters for 3 centers (2 params per center in a reduced 2D space: temp, pH)
    - 1 parameter for the Gaussian width (sigma)
    - 3 parameters for the weights of the 3 Gaussian basis functions
    
    The inputs b and s are combined into a single nutrient signal (log average) 
    to reduce dimensionality, as they likely have similar effects.
    The model is: y = sum_i w_i * exp(-||x - c_i||^2 / (2 * sigma^2))
    where x = [log(b), log(s), temp, pH] are normalized.
    
    This allows BFGS to fit the centers, width, and weights jointly, 
    adapting to the true structure of the growth rate surface.
    """
    # Normalize inputs
    # b, s: [1, 5] -> log scale to [0, 1]
    b_log = np.log(np.maximum(b, 1e-6))
    s_log = np.log(np.maximum(s, 1e-6))
    b_norm = (b_log - 0.0) / np.log(5.0)
    s_norm = (s_log - 0.0) / np.log(5.0)
    
    # temp: [1, 40] -> [0, 1]
    temp_norm = (temp - 1.0) / 39.0
    
    # pH: [2, 12] -> [0, 1]
    ph_norm = (pH - 2.0) / 10.0
    
    # Combine b and s into a single nutrient signal (geometric mean)
    nutrient = 0.5 * (b_norm + s_norm)
    
    # Feature vector: [nutrient, temp_norm, ph_norm]
    x = np.stack([nutrient, temp_norm, ph_norm], axis=-1)  # Shape (N, 3)
    
    # Extract parameters
    # params[0:6] = centers (3 centers x 2 params each? No, 3 centers x 3 dims = 9 params)
    # Wait, we have 10 params total. Let's use:
    # 3 centers x 3 dims = 9 params for centers
    # 1 param for sigma
    # But then we have no weights! 
    # Let's restructure: 
    # 2 centers x 3 dims = 6 params for centers
    # 1 param for sigma
    # 3 params for weights (one per center)
    # Total: 10 params. This is better.
    
    # Extract centers (2 centers, 3 dims each)
    c1 = params[0:3]
    c2 = params[3:6]
    
    # Extract sigma
    sigma = np.maximum(params[6], 0.01)  # Ensure positive width
    
    # Extract weights
    w1 = params[7]
    w2 = params[8]
    w3 = params[9]  # Use a third weight for a fixed center at the origin
    
    # Define fixed center at origin for the third basis function
    c3 = np.array([0.0, 0.0, 0.0])
    
    # Compute distances
    diff1 = x - c1[None, :]
    dist1_sq = np.sum(diff1 ** 2, axis=1)
    
    diff2 = x - c2[None, :]
    dist2_sq = np.sum(diff2 ** 2, axis=1)
    
    diff3 = x - c3[None, :]
    dist3_sq = np.sum(diff3 ** 2, axis=1)
    
    # Compute Gaussian basis values
    sigma_sq = 2.0 * sigma * sigma
    phi1 = np.exp(-dist1_sq / sigma_sq)
    phi2 = np.exp(-dist2_sq / sigma_sq)
    phi3 = np.exp(-dist3_sq / sigma_sq)
    
    # Weighted sum
    y = w1 * phi1 + w2 * phi2 + w3 * phi3
    
    return y
