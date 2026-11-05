import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

def construct():
    # Constants
    s = 1.0
    sqrt3 = np.sqrt(3)
    
    # Centers of the 11 hexagons
    # Using Pointy-Top orientation
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0.0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5),
        (0.0, 3.0),
        (0.0, -3.0),
        (2*sqrt3, 0.0),
        (-2*sqrt3, 0.0)
    ]
    
    # Generate vertices for all hexagons
    # Pointy-Top hexagon vertices relative to center
    # Angles: 30, 90, 150, 210, 270, 330 degrees?
    # Standard Pointy-Top:
    # (0, 1), (sqrt3/2, 0.5), (sqrt3/2, -0.5), (0, -1), (-sqrt3/2, -0.5), (-sqrt3/2, 0.5)
    rel_vertices = np.array([
        [0.0, 1.0],
        [sqrt3/2, 0.5],
        [sqrt3/2, -0.5],
        [0.0, -1.0],
        [-sqrt3/2, -0.5],
        [-sqrt3/2, 0.5]
    ])
    
    all_points = []
    for cx, cy in centers:
        for vx, vy in rel_vertices:
            all_points.append([cx + vx, cy + vy])
            
    all_points = np.array(all_points)
    
    # Compute Convex Hull
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]
    
    # Function to calculate required side length L for a given rotation alpha
    def get_L(alpha):
        # Normals of the enclosing hexagon
        # A regular hexagon with rotation alpha has normals at alpha, alpha+pi/3, ...
        # The distance from center to side is d = L * sqrt(3) / 2
        # So L = 2 * d / sqrt(3)
        # d_max = max |p . n| for all p in hull_points
        max_proj = 0.0
        for k in range(6):
            angle = alpha + k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            # Projection of all hull points onto this normal
            projections = np.abs(hull_points[:, 0] * nx + hull_points[:, 1] * ny)
            max_proj = max(max_proj, np.max(projections))
        
        return 2.0 * max_proj / sqrt3

    # Optimize alpha
    # Due to symmetry of the cluster (D2 symmetry? No, D4? No, just X and Y symmetry),
    # the function is periodic with pi/3.
    # We can search in [0, pi/6] (0 to 30 degrees)
    res = minimize_scalar(get_L, bounds=(0, np.pi/6), method='bounded')
    optimal_alpha = res.x
    optimal_L = res.fun
    
    # Prepare output
    # Inner hexagons: (x, y, angle_degrees)
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    center = (0.0, 0.0)
    side = float(optimal_L)
    angle = float(np.degrees(optimal_alpha))
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We just need to ensure it's the entry point.
# The task says "Keep the function names and signatures of the seed program."
# The seed program had `def construct():` inside the evolve block?
# No, the seed program had:
# """Initial program..."""
# N = 11
# # EVOLVE-BLOCK-START
# def construct():
#     ...
# # EVOLVE-BLOCK-END
#
# So I should put my code in the evolve block or replace the whole file?
# "Rewrite the program from scratch... Output the COMPLETE program in a single python code block."
# So I will output the full file.
# EVOLVE-BLOCK-END
