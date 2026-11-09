import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for a 3x4 hexagonal grid
    # Axial coordinates: q, r
    # Convert to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # We want a 3 (width) x 4 (height) block.
    # Let's iterate r from 0 to 3, q from 0 to 2.
    centers = []
    for r in range(4):
        for q in range(3):
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the grid at the origin
    center_of_mass = np.mean(centers, axis=0)
    centers -= center_of_mass
    
    # The outer hexagon is fixed. Let's say its normals are at 0, 60, 120 degrees.
    # Normal vectors:
    n0 = np.array([1.0, 0.0])
    n1 = np.array([np.cos(np.pi/3), np.sin(np.pi/3)])
    n2 = np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
    normals = np.vstack([n0, n1, n2])
    
    # Function to calculate L for a given rotation angle theta of the inner cluster
    def calc_L(theta):
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        
        # Project onto normals
        # projections[i, j] is projection of center i onto normal j
        projections = rotated_centers @ normals.T
        
        # Max absolute projection
        max_proj = np.max(np.abs(projections))
        
        # L calculation:
        # L * sqrt(3) / 2 >= max_proj + sqrt(3) / 2
        # L >= (2/sqrt(3)) * (max_proj + sqrt(3)/2)
        L = (2.0 / np.sqrt(3.0)) * (max_proj + np.sqrt(3.0) / 2.0)
        return L

    # Optimize theta
    # Due to symmetry, we only need to search 0 to pi/6 (30 degrees)
    result = minimize_scalar(calc_L, bounds=(0, np.pi/6), method='bounded')
    best_theta = result.x
    best_L = result.fun
    
    # Apply the best rotation to get final inner positions
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    # Construct the return list
    # inner rows: (x, y, angle_degrees)
    # Angle of inner hexagons: 0 (aligned with outer)
    inner = [[float(x), float(y), 0.0] for x, y in final_centers]
    
    # Outer hexagon: center (0,0), side best_L, angle 0
    return inner, (0.0, 0.0), float(best_L), 0.0
# EVOLVE-BLOCK-END
