"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 to maximize sum of radii.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    """
    best_sum = 0
    best_result = None
    
    # The rectangle has perimeter 4, so 2*(w+h) = 4, i.e., w+h = 2.
    # We'll try different aspect ratios and grid configurations.
    
    # Try different widths (height = 2 - width)
    for w in np.linspace(0.3, 1.7, 80):
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Try different grid configurations
        for cols in range(1, 15):
            rows = int(np.ceil(N / cols))
            if rows * cols < N:
                continue
            if rows > 20:  # Limit to reasonable grids
                continue
            
            # Grid cell sizes
            dx = w / cols
            dy = h / rows
            
            # Maximum radius for circles in this grid
            r = min(dx / 2.0, dy / 2.0)
            
            if r <= 0:
                continue
            
            # Sum of radii
            total_r = N * r
            
            if total_r > best_sum:
                best_sum = total_r
                # Create circle positions
                result = []
                for i in range(N):
                    col = i % cols
                    row = i // cols
                    x = (col + 0.5) * dx
                    y = (row + 0.5) * dy
                    result.append([x, y, r])
                best_result = np.array(result)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
