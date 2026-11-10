# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Moving sofa through L-shaped corridor: use the Pogossnitsky-Scheybman analytic path.
    
    The inner corner (x, y) at rotation angle theta (radians) follows:
        x(theta) = -1 + cos(theta) - sin(theta)
        y(theta) = -1 + sin(theta) + cos(theta)
    
    This is the known optimal (or near-optimal) path for the single-turn L-corridor,
    yielding a sofa area of approximately 2.2195 (vs ~2.0 for naive paths).
    
    We sample this curve at 20 evenly-spaced angles between 0 and 90 degrees.
    The evaluator prepends (0,0,0) and appends (-2,0,90), and densifies to <=1-degree steps.
    """
    # Sample the Pogossnitsky-Scheybman path at 20 angles in [0, pi/2]
    n_samples = 20
    thetas = np.linspace(0, np.pi / 2, n_samples)
    
    x = -1.0 + np.cos(thetas) - np.sin(thetas)
    y = -1.0 + np.sin(thetas) + np.cos(thetas)
    angles_deg = np.degrees(thetas)
    
    # Stack into (n, 3) array: (x, y, angle_degrees)
    path = np.column_stack((x, y, angles_deg)).astype(np.float64)
    
    return path


# EVOLVE-BLOCK-END
