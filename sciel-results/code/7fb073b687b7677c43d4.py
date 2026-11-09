import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Convert to float64 array
    x = np.asarray(x, dtype=np.float64)
    
    # Handle potential edge cases
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    # Define 10 evenly spaced knots across [0.1, 6.0]
    x_min, x_max = 0.1, 6.0
    knots = np.linspace(x_min, x_max, 10)
    
    # Compute hat functions (piecewise linear basis)
    # For each knot i, the hat function is:
    #   0 for x <= knots[i-1] (or x <= x_min for i=0)
    #   (x - knots[i-1]) / (knots[i] - knots[i-1]) for knots[i-1] < x < knots[i]
    #   1 at x = knots[i]
    #   (knots[i+1] - x) / (knots[i+1] - knots[i]) for knots[i] < x < knots[i+1]
    #   0 for x >= knots[i+1] (or x >= x_max for i=9)
    
    # For interior knots (1 to 8): standard hat
    # For boundary knots (0 and 9): half-hat (ramp up/down)
    
    basis = np.zeros((n, 10), dtype=np.float64)
    
    for i in range(10):
        if i == 0:
            # Left boundary: ramp up from knots[0] to knots[1]
            x_left = knots[0]
            x_right = knots[1]
            width = x_right - x_left
            # Clamp x to [x_left, x_right] for the ramp
            x_clamped = np.clip(x, x_left, x_right)
            basis[:, i] = (x_clamped - x_left) / width
            # But we only want it nonzero in [x_left, x_right]
            # Since x >= x_min = knots[0], and we clip, this handles it
            # Actually for x < x_left, (x - x_left)/width would be negative, but clipped to 0
            # For x > x_right, clipped to x_right, so (x_right - x_left)/width = 1
            # We want it to be 0 for x > x_right, so let's use a different approach
            basis[:, i] = np.where(x >= x_left, np.minimum((x - x_left) / width, 1.0), 0.0)
            # Actually, for x > x_right, we want 0. Let me fix:
            basis[:, i] = np.where((x >= x_left) & (x <= x_right), (x - x_left) / width, 0.0)
            
        elif i == 9:
            # Right boundary: ramp down from knots[8] to knots[9]
            x_left = knots[8]
            x_right = knots[9]
            width = x_right - x_left
            basis[:, i] = np.where((x >= x_left) & (x <= x_right), (x_right - x) / width, 0.0)
            
        else:
            # Interior knot: standard hat function
            x_left = knots[i-1]
            x_center = knots[i]
            x_right = knots[i+1]
            left_width = x_center - x_left
            right_width = x_right - x_center
            
            # Left side: (x - x_left) / left_width for x_left <= x <= x_center
            # Right side: (x_right - x) / right_width for x_center <= x <= x_right
            left_val = np.where((x >= x_left) & (x <= x_center), (x - x_left) / left_width, 0.0)
            right_val = np.where((x >= x_center) & (x <= x_right), (x_right - x) / right_width, 0.0)
            basis[:, i] = left_val + right_val
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
