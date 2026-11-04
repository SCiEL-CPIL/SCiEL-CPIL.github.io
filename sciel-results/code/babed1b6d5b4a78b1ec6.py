import numpy as np
from scipy.spatial import ConvexHull

def construct():
    N = 11
    
    # Helper to compute minimal enclosing hexagon side for a set of centers
    def min_enclosing_hex_side(centers):
        centers = np.array(centers)
        if len(centers) == 1:
            return 1.0
            
        # Convex hull of centers
        hull = ConvexHull(centers)
        hull_vertices = centers[hull.vertices]
        
        # Unit hexagon apothem
        a1 = np.sqrt(3) / 2
        
        # Function to compute support function of the Minkowski sum S = H_C + H_1
        # h_S(phi) = h_HC(phi) + h_H1(phi)
        def h_S(phi):
            # h_HC(phi)
            # Max dot product of hull vertices with unit vector (cos phi, sin phi)
            u_vec = np.array([np.cos(phi), np.sin(phi)])
            h_c = np.max(hull_vertices @ u_vec)
            
            # h_H1(phi)
            # Unit hexagon (radius 1, apothem sqrt(3)/2)
            # Symmetry period pi/3
            phi_mod = phi % (np.pi / 3)
            # Shift so that 0 is at the normal to a face
            # Standard hexagon with flat top has normals at 0, 60, ...
            # Actually, let's define H1 with a vertex at (1,0).
            # Normals are at 30, 90, ...
            # Let's stick to the formula: h1(phi) = a1 / cos(delta)
            # where delta is angle from nearest face normal.
            # Face normals of unit hex (vertex at 1,0) are at 30, 90, 150...
            # Let's assume standard orientation where one face is horizontal (normal 0).
            # Then vertices are at 30, 90...
            # Let's use the property that h1(phi) is the max distance from origin to face.
            # For a regular hexagon with apothem a, h(phi) = a / cos(phi - phi_face)
            # We can just compute the max projection of the unit hex vertices.
            # Vertices of unit hex (radius 1):
            hex_verts = np.array([
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2],
                [-0.5, np.sqrt(3)/2],
                [-1.0, 0.0],
                [-0.5, -np.sqrt(3)/2],
                [0.5, -np.sqrt(3)/2]
            ])
            h_1 = np.max(hex_verts @ u_vec)
            
            return h_c + h_1

        # Minimize L(theta) = (2/sqrt(3)) * max_k h_S(theta + k*pi/3)
        # Due to symmetry, we only need to search theta in [0, pi/6]
        min_L = float('inf')
        # High resolution search
        thetas = np.linspace(0, np.pi/6, 1000)
        for theta in thetas:
            max_h = 0
            for k in range(6):
                phi = theta + k * np.pi / 3
                h = h_S(phi)
                if h > max_h:
                    max_h = h
            L = (2.0 / np.sqrt(3)) * max_h
            if L < min_L:
                min_L = L
                
        return min_L

    # Candidate 1: Linear Row
    centers_lin = [(k * np.sqrt(3), 0) for k in range(-5, 6)]
    L_lin = min_enclosing_hex_side(centers_lin)
    
    # Candidate 2: 3x4 Rhombus (minus 1 corner)
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    centers_rh = []
    for i in range(3):
        for j in range(4):
            if not (i == 2 and j == 3): # Remove one corner
                p = i * u + j * v
                centers_rh.append(p)
                
    L_rh = min_enclosing_hex_side(centers_rh)
    
    # Candidate 3: 7-cluster + 4 extensions
    # 7-cluster centers
    c7 = [(0,0)]
    for k in range(6):
        ang = k * np.pi / 3
        c7.append((np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang)))
        
    # Add 4 more. Where?
    # Extend along the axes of the 7-cluster?
    # The 7-cluster has "gaps" at the corners.
    # Let's just use the Rhombus if it's better.
    
    best_L = min(L_lin, L_rh)
    best_centers = centers_rh if L_rh < L_lin else centers_lin
    
    # Construct the inner list
    # The problem asks for (x, y, angle_degrees)
    # We assume all inner hexagons are rotated 0 degrees relative to the global frame?
    # "each inner hexagon may translate and rotate"
    # If we rotate the inner hexagons, the shape S changes.
    # However, for a lattice packing, keeping them aligned is usually optimal or near-optimal.
    # Let's assume angle 0 for all.
    
    inner = []
    for p in best_centers:
        inner.append([p[0], p[1], 0.0])
        
    # Center of the outer hexagon
    # The optimal outer hexagon center is the centroid of the hull?
    # Or the center of the minimal enclosing hexagon?
    # The function min_enclosing_hex_side assumed the outer hexagon is centered at (0,0).
    # So we should return center (0,0) if the cluster is centered at 0.
    # For the rhombus, the cluster is not centered at 0.
    # We should shift the cluster so its "center" is at 0.
    
    # Let's re-center the best cluster
    centroid = np.mean(best_centers, axis=0)
    shifted_centers = [p - centroid for p in best_centers]
    
    # Recalculate L for shifted centers?
    # The relative geometry is the same, so L is the same.
    # But the outer hexagon center should be the centroid?
    # Actually, the minimal enclosing hexagon's center is the center of symmetry of the optimal orientation.
    # For simplicity, we return the centroid as the center.
    
    inner = []
    for p in shifted_centers:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-START
def construct():
    N = 11
    
    def min_enclosing_hex_side(centers):
        centers = np.array(centers)
        if len(centers) == 1:
            return 1.0
            
        hull = ConvexHull(centers)
        hull_vertices = centers[hull.vertices]
        
        hex_verts = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

        def h_S(phi):
            u_vec = np.array([np.cos(phi), np.sin(phi)])
            h_c = np.max(hull_vertices @ u_vec)
            h_1 = np.max(hex_verts @ u_vec)
            return h_c + h_1

        min_L = float('inf')
        thetas = np.linspace(0, np.pi/6, 2000)
        for theta in thetas:
            max_h = 0
            for k in range(6):
                phi = theta + k * np.pi / 3
                h = h_S(phi)
                if h > max_h:
                    max_h = h
            L = (2.0 / np.sqrt(3)) * max_h
            if L < min_L:
                min_L = L
                
        return min_L

    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    centers_rh = []
    for i in range(3):
        for j in range(4):
            if not (i == 2 and j == 3):
                p = i * u + j * v
                centers_rh.append(p)
                
    L_rh = min_enclosing_hex_side(centers_rh)
    
    centroid = np.mean(centers_rh, axis=0)
    shifted_centers = [p - centroid for p in centers_rh]
    
    inner = []
    for p in shifted_centers:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), L_rh, 0.0
# EVOLVE-BLOCK-END
