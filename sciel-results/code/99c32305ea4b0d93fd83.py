import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # The optimal configuration for points on a circle to maximize the ratio
    # of the minimum triangle area to the convex hull area is the regular polygon.
    # We construct the vertices of a regular 14-gon on the unit circle.
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
