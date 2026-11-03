"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        area = triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def potential_energy(points):
    """
    Smooth objective function to minimize.
    We minimize the sum of inverse squared areas of all triangles.
    This encourages all areas to be large, thereby increasing the minimum.
    """
    n = len(points)
    energy = 0.0
    for i, j, k in combinations(range(n), 3):
        area = triangle_area(points[i], points[j], points[k])
        if area < 1e-6:
            return 1e6 # Huge penalty for degenerate triangles
        energy += 1.0 / (area * area)
    return energy

def objective_function(x):
    """
    Objective function for scipy.optimize.minimize.
    x is a flat array of length 2*N.
    """
    points = x.reshape((N, 2))
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            # Calculate distance to nearest edge or just use a large penalty
            # Simple penalty: sum of squared distances to vertices if outside?
            # Better: check barycentric coordinates
            # Let's use a soft constraint penalty
            # Calculate distance to the closest point on the triangle boundary
            # For simplicity, if outside, add a large penalty based on how far out
            # We can project to the triangle or just penalize.
            # Let's calculate the "outside-ness"
            # Using the sign function from point_in_triangle logic
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
            d1 = sign(p, V1, V2)
            d2 = sign(p, V2, V3)
            d3 = sign(p, V3, V1)
            
            # The area of the triangle is 0.5 * sqrt(3)/2 = sqrt(3)/4
            # The values d1, d2, d3 are related to barycentric coords.
            # If any d is negative, it's outside.
            # We penalize the square of the negative part.
            penalty += max(0, -d1)**2 + max(0, -d2)**2 + max(0, -d3)**2
            
    if penalty > 1e-4:
        return 1e6 + penalty # Large penalty for being outside
    
    return potential_energy(points)

def initial_configuration_lattice():
    """
    Generate a structured initial configuration using a triangular lattice.
    """
    points = []
    # We want 11 points.
    # A triangular lattice with spacing s.
    # Let's try a grid of rows.
    # Row 0: 1 point? No, let's distribute.
    # Known good distributions for small N often look like:
    # N=1: Centroid
    # N=4: Vertices + Centroid
    # N=9: 3x3 grid (adapted)
    # N=11: 3x3 grid + 2 extra? Or 4-4-3 rows?
    
    # Let's use a 4-4-3 row structure but optimized positions.
    # Row 0 (y=0.1): 4 points
    # Row 1 (y=0.4): 4 points
    # Row 2 (y=0.7): 3 points
    
    # Better: Use a triangular lattice pattern.
    # Let's place points on a triangular grid.
    # Spacing h.
    # Points: (i*h, j*h*sqrt(3)) + offset
    
    # Let's just use the previous initial config but slightly better spaced,
    # or generate a lattice.
    
    # Lattice generation:
    # Let's try to fit a triangular lattice.
    # Side length of big triangle is 1.
    # Let's try 3 rows of lattice points.
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 4 points
    # Row 2 (top): 3 points
    
    # Coordinates for a triangular lattice with unit side length 1:
    # y_k = k * sqrt(3)/2
    # x_i = i/2 + (k % 2) * 0.5  (shifted rows)
    
    # We need to scale this to fit inside the triangle.
    # The triangle has height sqrt(3)/2.
    # If we use 3 rows, the y-coords could be 0.1, 0.4, 0.7.
    
    # Let's construct a specific good starting point.
    # Based on literature, for N=11, a good config is often close to a triangular lattice.
    
    # Let's use a 3-4-4 distribution?
    # Let's stick to the 4-4-3 but with better x-coordinates.
    
    y0 = 0.1
    y1 = 0.4
    y2 = 0.7
    
    # Row 0: 4 points, x from 0.15 to 0.85
    x0s = np.linspace(0.15, 0.85, 4)
    for x in x0s:
        points.append([x, y0])
        
    # Row 1: 4 points, x from 0.25 to 0.75
    x1s = np.linspace(0.25, 0.75, 4)
    for x in x1s:
        points.append([x, y1])
        
    # Row 2: 3 points, x from 0.35 to 0.65
    x2s = np.linspace(0.35, 0.65, 3)
    for x in x2s:
        points.append([x, y2])
        
    return np.array(points)

def construct():
    np.random.seed(42)
    
    # Get initial configuration
    points = initial_configuration_lattice()
    
    # Flatten for optimizer
    x0 = points.flatten()
    
    # Use Nelder-Mead for derivative-free optimization
    # We minimize the potential energy
    # Method='Nelder-Mead' is robust and doesn't require gradients
    
    # We run the optimizer
    # maxiter is set to allow sufficient convergence
    # tol for stopping
    
    result = minimize(
        objective_function, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
    )
    
    best_points = result.x.reshape((N, 2))
    
    # Verify and potentially do a second pass if needed
    # Check if points are valid
    valid = all(point_in_triangle(p) for p in best_points)
    
    if not valid:
        # Fallback to initial if optimizer went outside
        best_points = points
    else:
        # Calculate score
        current_score = min_triangle_area(best_points)
        
        # Optional: Run a second optimization from this point with tighter tolerances
        # Or just return. Nelder-Mead is usually good enough.
        
        # Let's try one more refinement pass
        x1 = best_points.flatten()
        result2 = minimize(
            objective_function, 
            x1, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        best_points2 = result2.x.reshape((N, 2))
        valid2 = all(point_in_triangle(p) for p in best_points2)
        
        if valid2:
            score2 = min_triangle_area(best_points2)
            if score2 > current_score:
                best_points = best_points2
            else:
                best_points = best_points # Keep the previous one if worse
        
    return best_points
# EVOLVE-BLOCK-END
