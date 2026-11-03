import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def hex_vertices(cx, cy, angle_deg=0.0):
    """Return vertices of a unit regular hexagon centered at (cx, cy) with given angle."""
    # Unit hexagon vertices relative to center (pointy-top, angle 0 means vertex at top? 
    # Let's use standard math: angle 0 is vertex at (1,0) -> flat-top? 
    # Task says "angle_degrees". 
    # Let's assume angle 0 means a vertex is at (1,0) (flat-top) or (0,1) (pointy-top).
    # It doesn't matter for the inner packing as long as we are consistent, 
    # but it affects the bounding box calculation.
    # Let's use pointy-top for the lattice generation, then rotate if needed.
    # Pointy-top vertices at angles 30, 90, 150, 210, 270, 330.
    # Or 0, 60, 120...
    # Let's use 0, 60, 120, 180, 240, 300 for simplicity (flat-top).
    angles = np.deg2rad(np.arange(0, 360, 60) + angle_deg)
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def axial_to_cartesian(q, r):
    # Pointy-top axial to cartesian
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def get_cluster_vertices(cluster_axial, angle_deg=0.0):
    """Get all vertices for a cluster defined by axial coordinates."""
    verts = []
    for q, r in cluster_axial:
        x, y = axial_to_cartesian(q, r)
        # Inner hexagons are pointy-top in this coordinate system?
        # If we use axial_to_cartesian for pointy-top, the hexagons themselves
        # should be aligned with the grid.
        # A pointy-top hexagon has vertices at 30, 90, ...
        # Let's generate vertices for a pointy-top hexagon at (x,y).
        angles = np.deg2rad(np.arange(30, 390, 60) + angle_deg)
        hx = x + np.cos(angles)
        hy = y + np.sin(angles)
        verts.append(np.column_stack((hx, hy)))
    return np.vstack(verts)

def min_outer_side(vertices, theta_deg):
    """Calculate min outer hexagon side length for given vertices and outer rotation."""
    theta = np.deg2rad(theta_deg)
    # Normals of outer hexagon (flat-top reference)
    # If outer is flat-top, normals are at 0, 60, 120, 180, 240, 300.
    # If rotated by theta, normals are at theta, theta+60, ...
    normal_angles = np.deg2rad(np.arange(0, 360, 60) + theta)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Project vertices onto normals
    # vertices shape (M, 2), normals shape (6, 2)
    projections = vertices @ normals.T # (M, 6)
    
    widths = []
    for k in range(6):
        w = np.max(projections[:, k]) - np.min(projections[:, k])
        widths.append(w)
    
    max_width = max(widths)
    # Side length L such that width = L * sqrt(3)
    return max_width / np.sqrt(3)

def evaluate_cluster(cluster_axial, inner_angle=0.0):
    """Optimize outer rotation for a given cluster."""
    vertices = get_cluster_vertices(cluster_axial, inner_angle)
    
    # Objective function
    def obj(theta):
        return min_outer_side(vertices, theta)
        
    # Search for minimum in [0, 60]
    res = minimize_scalar(obj, bounds=(0, 60), method='bounded')
    best_theta = res.x
    best_side = res.fun
    
    # Construct inner list
    inner = []
    for q, r in cluster_axial:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, inner_angle])
        
    return inner, (0.0, 0.0), best_side, best_theta

# EVOLVE-BLOCK-START
def construct():
    # Define candidate clusters
    # Cluster 1: Flower (7) + 5
    c1 = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0)
    ]
    
    # Cluster 2: 3-4-5
    c2 = []
    # r=0, q=-2..2 (5)
    for q in range(-2, 3): c2.append((q, 0))
    # r=1, q=-1..2 (4)
    for q in range(-1, 3): c2.append((q, 1))
    # r=-1, q=-2..0 (3)
    for q in range(-2, 1): c2.append((q, -1))
    
    # Cluster 3: 2-4-4-2
    c3 = []
    # r=1, q=-1..0 (2)
    for q in range(-1, 1): c3.append((q, 1))
    # r=0, q=-1..2 (4)
    for q in range(-1, 3): c3.append((q, 0))
    # r=-1, q=-2..1 (4)
    for q in range(-2, 2): c3.append((q, -1))
    # r=-2, q=-1..0 (2)
    for q in range(-1, 1): c3.append((q, -2))
    
    # Cluster 4: 1-3-4-4 (Top heavy?)
    c4 = []
    for q in range(0, 1): c4.append((q, 2))
    for q in range(-1, 2): c4.append((q, 1))
    for q in range(-2, 2): c4.append((q, 0))
    for q in range(-2, 2): c4.append((q, -1))
    
    candidates = [c1, c2, c3, c4]
    
    best_result = None
    best_side = float('inf')
    
    for c in candidates:
        # Try a few inner angles just in case (though 0 is likely best for honeycomb)
        for angle in [0.0, 30.0]:
            inner, center, side, theta = evaluate_cluster(c, angle)
            if side < best_side:
                best_side = side
                best_result = (inner, center, side, theta)
                
    return best_result
# EVOLVE-BLOCK-END
