import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def construct():
    # Define lattice vectors for hexagonal packing of unit hexagons
    # Distance between centers of adjacent unit hexagons is sqrt(3)
    d = np.sqrt(3)
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, 1.5])  # 60 degrees from u
    
    # Define the 3x4 grid of centers
    # i in 0..2, j in 0..3
    # Center the grid at origin for symmetry
    centers = []
    for i in range(3):
        for j in range(4):
            # Shift to center: i=1 is center of 0,1,2; j=1.5 is center of 0,1,2,3
            ci = i - 1.0
            cj = j - 1.5
            c = ci * u + cj * v
            centers.append(c)
    
    centers = np.array(centers)
    
    # The centroid is (0,0) due to symmetry of the grid indices
    # i: 0,1,2 -> -1, 0, 1. Sum = 0.
    # j: 0,1,2,3 -> -1.5, -0.5, 0.5, 1.5. Sum = 0.
    center_of_mass = np.sum(centers, axis=0) / N
    
    # Function to calculate width of a set of points in direction theta
    def width_centers(theta):
        n = np.array([np.cos(theta), np.sin(theta)])
        projections = centers @ n
        return np.max(projections) - np.min(projections)
        
    # Function to calculate width of a single unit regular hexagon in direction theta
    # Vertices at angles k*pi/3
    def width_inner(theta):
        # W(theta) = 2 * max_k cos(theta - k*pi/3)
        # Due to symmetry, we can just check k=0..2 and use abs?
        # No, max over all 6.
        max_proj = 0.0
        for k in range(6):
            angle_diff = theta - k * np.pi / 3.0
            proj = np.cos(angle_diff)
            if proj > max_proj:
                max_proj = proj
        return 2.0 * max_proj
        
    # Objective function: Minimize S(theta)
    # S(theta) = max_k (W_centers(theta + k*pi/3) + W_inner(theta + k*pi/3)) / sqrt(3)
    def objective(theta):
        max_width = 0.0
        for k in range(3):
            ang = theta + k * np.pi / 3.0
            w_c = width_centers(ang)
            w_i = width_inner(ang)
            total_width = w_c + w_i
            if total_width > max_width:
                max_width = total_width
        return max_width / np.sqrt(3)
        
    # Optimize theta in [0, pi/6]
    # The function is periodic with pi/3, and symmetric around 0?
    # Let's just minimize in [0, pi/6]
    res = minimize_scalar(objective, bounds=(0, np.pi/6), method='bounded')
    best_theta = res.x
    
    # Calculate final side length
    best_side = objective(best_theta)
    
    # Construct inner hexagons
    # They are at centers, with angle 0
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Outer hexagon center
    outer_center = [center_of_mass[0], center_of_mass[1]]
    
    # Outer hexagon angle
    outer_angle = best_theta
    
    return inner, outer_center, best_side, outer_angle
