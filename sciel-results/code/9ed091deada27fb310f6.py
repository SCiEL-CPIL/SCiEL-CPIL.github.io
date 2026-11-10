import numpy as np

def moving_sofa_2d() -> np.ndarray:
    """
    Constructs a path for the moving sofa problem using a dense geometric curve.
    This approach uses a semi-circular arc for the translation of the corridor's
    inner corner while rotating the corridor from 0 to 90 degrees.
    """
    
    # Number of intermediate points to define the path densely
    # The evaluator interpolates linearly between returned points.
    # More points = better approximation of the curve.
    n_points = 50
    
    # Generate points along a semi-circle from (0,0) to (-2,0)
    # Center (-1, 0), Radius 1
    # theta from 0 to pi
    thetas = np.linspace(0, np.pi, n_points)
    
    x_coords = -1.0 + np.cos(thetas)
    y_coords = np.sin(thetas)
    
    # Angle a from 0 to 90 degrees
    # We want to map the path progression to the angle.
    # Linear mapping of theta to angle?
    # Or maybe angle should change faster when the sofa is "tight"?
    # Let's try linear angle change with respect to path length or index.
    angles = np.linspace(0, 90, n_points)
    
    # The function must return the INTERMEDIATE points.
    # The evaluator adds (0,0,0) and (-2,0,90).
    # So we should exclude the first and last points of our generated array
    # if they match the endpoints, or just return the middle ones.
    # My generated points:
    # Index 0: theta=0 -> (0,0), angle=0. Matches start.
    # Index -1: theta=pi -> (-2,0), angle=90. Matches end.
    
    # So we return indices 1 to n_points-2
    path = np.column_stack((x_coords, y_coords, angles))
    
    # Exclude endpoints
    if len(path) > 2:
        return path[1:-1]
    else:
        # Fallback if n_points is too small
        return np.array([[ -1.0, 1.0, 45.0 ]])
