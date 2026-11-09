"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Fixed-Center Radial Basis Function (RBF) Network.
    
    Strategy:
    We model the growth rate as a linear combination of 10 Gaussian RBFs.
    The centers and widths of these RBFs are fixed based on the known input ranges.
    The 10 parameters to be fitted are the weights (amplitudes) of these RBFs.
    
    Why this works:
    1. Linearity in params: The function is linear with respect to `params`. 
       This makes the loss surface convex (or at least much smoother) with respect 
       to the parameters, avoiding the local minima traps of non-linear models.
    2. Expressiveness: Gaussian RBFs can approximate a wide variety of smooth 
       functions, capturing complex interactions between b, s, temp, and pH.
    3. Robustness: Since the centers are fixed and spread across the input domain, 
       the basis functions provide good coverage. BFGS starting from all-ones 
       will quickly converge to the optimal linear weights.
    
    Inputs:
    b: population density (range ~[1, 5])
    s: substrate (range ~[1, 40])
    temp: temperature (range ~[1, 12])
    pH: pH (range ~[1, 12])
    
    params: 10 constants representing the weights of the RBFs.
    """
    
    # Define the 10 fixed centers in the 4D input space.
    # We distribute them to cover the corners and center of the hypercube 
    # defined by the input ranges.
    # Ranges:
    # b: [1, 5] -> mid 3, half-width 2
    # s: [1, 40] -> mid 20.5, half-width 19.5
    # temp: [1, 12] -> mid 6.5, half-width 5.5
    # pH: [1, 12] -> mid 6.5, half-width 5.5
    
    # To ensure good coverage with only 10 centers, we use a mix of:
    # - Center of the space
    # - Centers of the 8 corners (low/high for each dimension)
    # This gives 1 + 8 = 9 centers. We add one more at a specific 
    # high-activity corner or just duplicate the center with a different width?
    # Better: Use 5 centers at the midpoints of the axes and 5 at corners?
    # Let's stick to a structured grid:
    # 2 points per dimension? 2^4 = 16 points, but we only have 10 params.
    # So we must select 10 specific points.
    
    # Let's use a "Latin Hypercube" style or specific key points.
    # Given the likely biology:
    # - Low b, Low s -> Low growth
    # - High b, High s -> High growth (up to a point)
    # - Optimal temp/pH -> High growth
    # - Extreme temp/pH -> Low growth
    
    # Let's place centers at:
    # 1. Global Center: (3, 20.5, 6.5, 6.5)
    # 2. Low-Low-Low-Low: (1, 1, 1, 1)
    # 3. High-High-High-High: (5, 40, 12, 12)
    # 4. High b, High s, Mid temp, Mid pH: (5, 40, 6.5, 6.5)
    # 5. Mid b, Mid s, High temp, Mid pH: (3, 20.5, 12, 6.5)
    # 6. Mid b, Mid s, Mid temp, High pH: (3, 20.5, 6.5, 12)
    # 7. Mid b, Mid s, Low temp, Mid pH: (3, 20.5, 1, 6.5)
    # 8. Mid b, Mid s, Mid temp, Low pH: (3, 20.5, 6.5, 1)
    # 9. High b, Low s, Mid temp, Mid pH: (5, 1, 6.5, 6.5)
    # 10. Low b, High s, Mid temp, Mid pH: (1, 40, 6.5, 6.5)
    
    centers = np.array([
        [3.0, 20.5, 6.5, 6.5],   # 0: Center
        [1.0, 1.0, 1.0, 1.0],    # 1: All Low
        [5.0, 40.0, 12.0, 12.0], # 2: All High
        [5.0, 40.0, 6.5, 6.5],   # 3: High b, High s
        [3.0, 20.5, 12.0, 6.5],  # 4: High temp
        [3.0, 20.5, 6.5, 12.0],  # 5: High pH
        [3.0, 20.5, 1.0, 6.5],   # 6: Low temp
        [3.0, 20.5, 6.5, 1.0],   # 7: Low pH
        [5.0, 1.0, 6.5, 6.5],    # 8: High b, Low s
        [1.0, 40.0, 6.5, 6.5]    # 9: Low b, High s
    ])
    
    # Define widths (sigma) for each RBF.
    # We want them to be wide enough to influence nearby regions but distinct.
    # Normalizing the inputs would help, but we can just set sigmas relative to the range.
    # b range 4, s range 39, temp range 11, pH range 11.
    # Let's use a standardized space internally for distance calculation to avoid scaling issues.
    
    # Normalize inputs to [0, 1] for distance calculation? 
    # Or just use the raw values with appropriate sigmas.
    # Using raw values:
    # sigma_b ~ 2, sigma_s ~ 10, sigma_temp ~ 3, sigma_pH ~ 3
    
    # Let's define sigmas per dimension for each center. 
    # To keep it simple and symmetric, we use a single width per RBF in a normalized space.
    
    # Normalize inputs to 0-1 range
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    t_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 11.0
    
    # Normalize centers to 0-1 range
    centers_norm = np.array([
        [0.5, 0.5, 0.5, 0.5],   # 0
        [0.0, 0.0, 0.0, 0.0],   # 1
        [1.0, 1.0, 1.0, 1.0],   # 2
        [1.0, 1.0, 0.5, 0.5],   # 3
        [0.5, 0.5, 1.0, 0.5],   # 4
        [0.5, 0.5, 0.5, 1.0],   # 5
        [0.5, 0.5, 0.0, 0.5],   # 6
        [0.5, 0.5, 0.5, 0.0],   # 7
        [1.0, 0.0, 0.5, 0.5],   # 8
        [0.0, 1.0, 0.5, 0.5]    # 9
    ])
    
    # Widths (sigma) in normalized space.
    # A sigma of 0.3 means the RBF has significant influence within ~0.3 units.
    # We vary the width slightly to allow for different locality.
    # Center (0) should be wide to capture the general trend.
    # Corners can be narrower.
    sigmas = np.array([
        0.5,  # 0: Center, wide
        0.3,  # 1: Corner
        0.3,  # 2: Corner
        0.4,  # 3: Edge
        0.4,  # 4: Edge
        0.4,  # 5: Edge
        0.4,  # 6: Edge
        0.4,  # 7: Edge
        0.4,  # 8: Edge
        0.4   # 9: Edge
    ])
    
    # Inputs shape (N, 4)
    inputs_norm = np.stack([b_norm, s_norm, t_norm, ph_norm], axis=1)
    
    # Calculate squared distances (N, 10)
    # dist_sq[i, j] = ||inputs_norm[i] - centers_norm[j]||^2
    # Efficient calculation:
    x_norm_sq = np.sum(inputs_norm**2, axis=1, keepdims=True) # (N, 1)
    k_norm_sq = np.sum(centers_norm**2, axis=1, keepdims=True).T # (1, 10)
    xk = inputs_norm @ centers_norm.T # (N, 10)
    
    dist_sq = x_norm_sq + k_norm_sq - 2 * xk
    # Ensure non-negative due to floating point errors
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Calculate RBF values
    # phi_ij = exp(-dist_sq_ij / (2 * sigma_j^2))
    sigmas_sq = sigmas**2
    # (10,) -> (1, 10) for broadcasting
    phi = np.exp(-dist_sq / (2.0 * sigmas_sq[np.newaxis, :]))
    
    # Prediction is the linear combination: sum_j phi_ij * params_j
    # phi is (N, 10), params is (10,)
    prediction = phi @ params
    
    return prediction
# EVOLVE-BLOCK-END
