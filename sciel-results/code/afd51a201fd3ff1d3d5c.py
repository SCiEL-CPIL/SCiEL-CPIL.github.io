import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for flat-top hexagons of side 1
    # Distance between centers is sqrt(3)
    sqrt3 = np.sqrt(3)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Define the 11 hexagon centers in lattice coordinates (i, j)
    # Shape: 1-3-3-3-1
    # j=0: i=0
    # j=1: i=-1, 0, 1
    # j=2: i=-1, 0, 1
    # j=3: i=-1, 0, 1
    # j=4: i=0
    lattice_coords = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    centers = []
    for i, j in lattice_coords:
        c = i * e1 + j * e2
        centers.append(c)
        
    centers = np.array(centers)
    
    # Center the cluster at (0,0)
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Generate vertices for all 11 hexagons
    # Flat-top hexagon vertices relative to center
    angles = np.deg2rad(np.arange(0, 360, 60))
    verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_verts = []
    for c in centers:
        verts = verts_rel + c
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # Function to calculate required side length for a given outer hexagon rotation
    def calc_side_length(theta):
        # Normals of the outer hexagon
        # Outer hexagon sides are perpendicular to normals
        # Normals are at theta + k * 60 deg
        normal_angles = theta + np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto normals
        # all_verts shape (66, 2), normals shape (6, 2)
        projections = all_verts @ normals.T
        
        # Max absolute projection is the apothem A
        A = np.max(np.abs(projections))
        
        # Side length S = 2 * A / sqrt(3)
        return 2.0 * A / sqrt3

    # Optimize theta
    # Due to symmetry of the cluster (approx) and hexagon, optimal theta is likely 0 or 30 deg
    # Scan 0 to 60 degrees
    thetas = np.linspace(0, np.pi / 3, 1000)
    sides = [calc_side_length(t) for t in thetas]
    
    min_side = np.min(sides)
    best_theta = thetas[np.argmin(sides)]
    
    # Prepare output
    # Inner hexagons: [x, y, angle_degrees]
    # We use angle 0 for all inner hexagons (aligned with lattice)
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Outer hexagon
    center = [0.0, 0.0]
    side = min_side
    angle_deg = np.rad2deg(best_theta)
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
