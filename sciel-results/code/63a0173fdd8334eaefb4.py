import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Use a piecewise linear basis (hat functions) with 10 fixed
    knots evenly spaced across the domain [0.1, 6]. This creates a continuous
    but non-smooth function class that can approximate discontinuities or sharp
    transitions which global polynomial and smooth RBF bases fail to capture,
    while remaining linear-in-parameters for stable BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 evenly spaced knots in [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    knots = np.linspace(x_min, x_max, 10)
    
    # Precompute distances to nearest neighbors for half-widths
    # For interior knots, width is distance to nearest knot
    # For boundary knots, width is distance to the only neighbor
    widths = np.zeros(10)
    for i in range(10):
        if i == 0:
            widths[i] = knots[1] - knots[0]
        elif i == 9:
            widths[i] = knots[9] - knots[8]
        else:
            # Use the smaller distance to ensure the hat function is zero at neighbors
            # Actually, for a standard hat function to be zero at *all* other knots,
            # the width must be exactly the distance to the nearest neighbor.
            # If we use max distance, it won't be zero at the further neighbor.
            # So we use min distance to left and right.
            widths[i] = min(knots[i] - knots[i-1], knots[i+1] - knots[i])
            
    # Compute basis functions
    # basis[i, j] is the value of the i-th hat function at x[j]
    # We can do this vectorized or loop. Loop is fine for 10 basis functions.
    basis = np.zeros((len(x), 10))
    
    for i in range(10):
        center = knots[i]
        width = widths[i]
        # h_i(x) = max(0, 1 - |x - center| / width)
        basis[:, i] = np.maximum(0.0, 1.0 - np.abs(x - center) / width)
        
    # Linear combination
    return basis @ params
