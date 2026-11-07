"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a Fixed Center Radial Basis Function (RBF) expansion.
    This approach uses 10 Gaussian basis functions centered at fixed points
    distributed across the input domain [-1, 1] x [-1, 1].
    The parameters are the weights for each basis function.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Define 10 fixed centers
    # Using a 5x2 grid: x in {-1, -0.5, 0, 0.5, 1}, y in {-1, 1}
    # This provides coverage across the domain.
    centers_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    centers_y = np.array([-1.0, 1.0])
    
    # Create meshgrid of centers
    cx, cy = np.meshgrid(centers_x, centers_y)
    cx = cx.flatten()
    cy = cy.flatten()
    
    # Gaussian width parameter (inverse bandwidth)
    gamma = 2.0
    
    pred = np.zeros_like(x)
    
    for i in range(10):
        # Calculate squared distance
        dist_sq = (x - cx[i])**2 + (y - cy[i])**2
        # Gaussian basis function
        basis_val = np.exp(-gamma * dist_sq)
        pred += p[i] * basis_val
        
    return pred
# EVOLVE-BLOCK-END
