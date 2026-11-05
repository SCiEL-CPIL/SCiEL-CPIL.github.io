import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 unit hexagon centers in a triangular lattice
    # Lattice vectors
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    # Create a 3x4 grid (12 hexagons) and remove one to get 11
    # This creates a compact cluster
    centers = []
    for i in range(3):
        for j in range(4):
            centers.append(i * u + j * v)
            
    # Remove the last one to make it 11
    centers = centers[:11]
    
    # Generate all vertices of the 11 hexagons
    # A unit hexagon has vertices at distance 1 from center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] # 0, 60, 120, 180, 240, 300
    vertices = []
    for c in centers:
        for a in angles:
            vertices.append(c + np.array([np.cos(a), np.sin(a)]))
            
    vertices = np.array(vertices)
    
    # Compute Convex Hull
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    
    # Function to compute the side length of the minimal enclosing regular hexagon
    # for a given rotation alpha
    def min_side_length(alpha):
        # Normal directions of the outer hexagon
        # The outer hexagon has sides perpendicular to these normals
        # Normals are at alpha, alpha + pi/3, alpha + 2*pi/3
        normals = []
        for k in range(3):
            theta = alpha + k * np.pi / 3
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        max_width = 0
        for n in normals:
            projections = hull_vertices @ n
            w = np.max(projections) - np.min(projections)
            if w > max_width:
                max_width = w
                
        # Side length L relates to width W by W = L * sqrt(3)
        # So L = W / sqrt(3)
        return max_width / np.sqrt(3)
        
    # Minimize min_side_length(alpha) for alpha in [0, pi/3]
    # Use ternary search
    lo = 0.0
    hi = np.pi / 3
    for _ in range(100):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if min_side_length(m1) < min_side_length(m2):
            hi = m2
        else:
            lo = m1
            
    alpha = (lo + hi) / 2
    side_length = min_side_length(alpha)
    
    # Determine the center of the outer hexagon
    # The center C should be such that the projections are centered
    # Ideally, C is the centroid of the hull vertices
    center = np.mean(hull_vertices, axis=0)
    
    # Construct the return values
    # Inner hexagons: (x, y, angle_degrees)
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    # Outer hexagon: center (x, y), side length, rotation (degrees)
    outer_center = [float(center[0]), float(center[1])]
    outer_side = float(side_length)
    outer_angle = float(np.degrees(alpha))
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
