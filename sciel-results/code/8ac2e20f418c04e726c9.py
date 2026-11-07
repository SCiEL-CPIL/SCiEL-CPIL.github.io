import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define knots for 10 cubic B-spline basis functions
    # Number of basis functions n = 10, degree k = 3
    # Number of knots m = n + k + 1 = 14? 
    # Standard definition: n basis functions require n+k+1 knots.
    # If we have knots t_0 ... t_{m-1}, and degree k, number of basis functions is m - k - 1.
    # We want 10 basis functions. 10 = m - 3 - 1 => m = 14.
    # So we need 14 knots.
    
    # Let's span [-3, 3].
    # To ensure the spline covers the range and has "clamped" or standard behavior,
    # we can distribute 14 knots evenly.
    knots = np.linspace(-3.0, 3.0, 14)
    
    # We need to compute N_i,3(x) for i=0 to 9.
    # N_i,3 is non-zero on [t_i, t_{i+4}].
    
    # Vectorized computation of basis functions
    # Initialize N_i,0
    # N_i,0(x) = 1 if t_i <= x < t_{i+1}
    
    n_basis = 10
    degree = 3
    
    # Create a matrix B of shape (len(x), n_basis)
    # We'll compute basis functions iteratively.
    
    # N[i] will store N_{i, j} for current degree j
    # Start with j=0
    # N_{i,0}(x)
    
    # Efficient way:
    # For each basis function i, we can compute its value directly using the formula
    # or use the recursive definition.
    # Given the small size, let's use the recursive definition vectorized over x.
    
    # N_{i,0}(x)
    # We need N_{i,1}, N_{i,2}, N_{i,3}.
    
    # Let's store the current level of basis functions in a list or array.
    # N_level_0[i] is N_{i,0}
    
    # Actually, it's easier to compute the basis for each x individually if N is small,
    # but N can be 200. Vectorizing over x is better.
    
    # Let's create an array `basis` of shape (10, len(x))
    # Initialize with N_{i,0}
    
    # N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
    # Note: For the last interval, usually defined differently or just <=.
    # For B-splines, the last knot is often handled to include the endpoint.
    
    # Let's use a standard recursive implementation.
    
    # N_i^0(x)
    N = np.zeros((10, len(x)))
    for i in range(10):
        # N_{i,0} is 1 in [knots[i], knots[i+1])
        # Using vectorized comparison
        mask = (x >= knots[i]) & (x < knots[i+1])
        # Handle the right endpoint for the last knot if necessary, 
        # but since we go up to knots[13]=3.0, and x<=3.0, 
        # the last basis function N_9,0 is in [knots[9], knots[10])? 
        # Wait, N_{i,0} uses knots[i] and knots[i+1].
        # Max i is 9. knots[9] to knots[10].
        # But we have 14 knots, indices 0..13.
        # Basis functions are defined for i=0..9.
        # N_{i,0} depends on knots[i] and knots[i+1].
        # So we need knots up to index 10 for N_9,0.
        # This is fine.
        
        N[i] = mask.astype(float)
        
    # Iterate for degree 1, 2, 3
    # We need to keep track of the previous level.
    # Let's use a 2D array for the current level and update it.
    # Actually, we need N_{i, j-1} and N_{i+1, j-1} to compute N_{i, j}.
    # So we can store all levels or just the previous one.
    
    # Let's store the previous level in `prev_N` and current in `curr_N`.
    # But we need to compute N_{i, j} for i=0..9.
    # N_{i, j} depends on N_{i, j-1} and N_{i+1, j-1}.
    # So we need N_{i+1, j-1} which is the next index in the previous level.
    
    # Let's re-initialize properly.
    # Level 0: N_{i,0} for i=0..12? 
    # To compute N_{0,3}, we need N_{0,0}, N_{1,0}, N_{2,0}, N_{3,0}.
    # To compute N_{9,3}, we need N_{9,0}, N_{10,0}, N_{11,0}, N_{12,0}.
    # So we need N_{i,0} for i=0..12.
    # We have 14 knots, so indices 0..13.
    # N_{i,0} is defined for i=0..12 (using knots[i] and knots[i+1]).
    
    # Let's compute N_{i,0} for i=0..12.
    N_prev = np.zeros((13, len(x))) # Indices 0..12
    for i in range(13):
        mask = (x >= knots[i]) & (x < knots[i+1])
        # For the very last point x=3.0, it might fall into the last bin if we use <= for the last one.
        # Standard B-spline definition: N_{n-1,0} includes the right endpoint.
        if i == 12:
             mask = (x >= knots[i]) & (x <= knots[i+1])
        N_prev[i] = mask.astype(float)
        
    # Now compute Level 1: N_{i,1} for i=0..11
    # N_{i,1} = (x - t_i)/(t_{i+1} - t_i) * N_{i,0} + (t_{i+2} - x)/(t_{i+2} - t_{i+1}) * N_{i+1,0}
    N_curr = np.zeros((12, len(x)))
    for i in range(12):
        denom1 = knots[i+1] - knots[i]
        denom2 = knots[i+2] - knots[i+1]
        term1 = (x - knots[i]) / denom1 * N_prev[i]
        term2 = (knots[i+2] - x) / denom2 * N_prev[i+1]
        N_curr[i] = term1 + term2
        
    # Level 2: N_{i,2} for i=0..10
    N_prev = N_curr
    N_curr = np.zeros((11, len(x)))
    for i in range(11):
        denom1 = knots[i+2] - knots[i]
        denom2 = knots[i+3] - knots[i+1]
        term1 = (x - knots[i]) / denom1 * N_prev[i]
        term2 = (knots[i+3] - x) / denom2 * N_prev[i+1]
        N_curr[i] = term1 + term2
        
    # Level 3: N_{i,3} for i=0..9
    N_prev = N_curr
    # Final basis functions are N_{0,3} ... N_{9,3}
    basis = np.zeros((10, len(x)))
    for i in range(10):
        denom1 = knots[i+3] - knots[i]
        denom2 = knots[i+4] - knots[i+1]
        term1 = (x - knots[i]) / denom1 * N_prev[i]
        term2 = (knots[i+4] - x) / denom2 * N_prev[i+1]
        basis[i] = term1 + term2
        
    # The prediction is sum(params[i] * basis[i])
    # params is shape (10,), basis is shape (10, N)
    # result = params @ basis -> shape (N,)
    
    return params @ basis
