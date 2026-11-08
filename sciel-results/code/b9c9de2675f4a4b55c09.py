import numpy as np
import itertools
import time

N = 11

def _hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle (degrees) and side length."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _min_enclosing_hexagon(vertices):
    """
    Find the minimum side length S and center (cx, cy) of a regular hexagon 
    that encloses the given vertices.
    Returns (S, cx, cy, phi_deg).
    """
    best_S = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0
    
    # Sweep angles from 0 to 60 degrees
    # The optimal rotation for a hexagon enclosing a set of points 
    # can be found by checking angles where a vertex of the inner set 
    # aligns with a normal of the outer hexagon.
    # However, a fine sweep is robust enough.
    
    centroid = np.mean(vertices, axis=0)
    
    for phi_deg in np.linspace(0, 60, 361): # Finer sweep
        a = np.radians(phi_deg)
        # Normals of the outer hexagon sides
        # The sides are perpendicular to the normals.
        # For a regular hexagon with angle phi, the normals are at phi + 30 + k*60
        normals = []
        for k in range(6):
            theta = a + np.radians(30 + k * 60)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals) # 6x2
        
        # Calculate the maximum projection distance from the centroid in each normal direction
        # If we center the outer hexagon at the centroid, the required apothem A is:
        # A = max_i,k | (v_i - centroid) . n_k |
        
        # Shift vertices to be relative to centroid
        verts_rel = vertices - centroid
        
        # Project onto normals
        # projections is Mx6
        projections = verts_rel @ normals.T 
        
        # We need the maximum absolute projection in each direction
        # max_abs_proj_k = max_i | proj_ik |
        max_abs_projs = np.max(np.abs(projections), axis=0) # 6
        
        # The required apothem is the maximum of these
        A = np.max(max_abs_projs)
        
        # Side length S = 2 * A / sqrt(3)
        S = 2.0 * A / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_phi = phi_deg
            best_cx = centroid[0]
            best_cy = centroid[1]

    # Add a small safety margin to ensure containment despite floating point errors
    best_S += 1e-5
    
    return best_S, best_cx, best_cy, best_phi

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Generate lattice centers
    # Ring 0
    centers = [(0.0, 0.0)]
    
    # Ring 1 (6 hexes)
    d = np.sqrt(3)
    ring1 = []
    for k in range(6):
        angle = k * np.pi / 3
        ring1.append((d * np.cos(angle), d * np.sin(angle)))
    centers.extend(ring1)
    
    # Ring 2 candidates
    # We need 4 more centers to make 11.
    # Current: 1 + 6 = 7.
    # Candidates: lattice points in the second shell.
    
    lattice_points = set()
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * d + j * d / 2
            y = j * d * np.sqrt(3) / 2
            # Exclude center and ring 1
            # Check distance to origin to identify ring 2
            dist_sq = x*x + y*y
            # Ring 1 dist_sq = 3
            # Ring 2 dist_sq = 9 (for 6 points) and 12 (for 6 points)?
            # Let's just collect points with dist_sq > 3.1 and < 13.0
            if 3.1 < dist_sq < 13.0:
                # Round to avoid float issues
                rx, ry = round(x, 10), round(y, 10)
                lattice_points.add((rx, ry))
    
    candidates = list(lattice_points)
    # Sort by distance to origin to pick the closest ones (Ring 2 is the first shell outside Ring 1)
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # We expect 12 candidates in the second shell.
    # If we have more, we take the first 12.
    ring2_candidates = candidates[:12]
    
    if len(ring2_candidates) < 4:
        # Fallback if lattice generation failed
        # Just place them in a line? No, must be valid.
        # Expand search range
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * d + j * d / 2
                y = j * d * np.sqrt(3) / 2
                dist_sq = x*x + y*y
                if 3.1 < dist_sq < 20.0:
                    rx, ry = round(x, 10), round(y, 10)
                    if (rx, ry) not in lattice_points:
                        lattice_points.add((rx, ry))
        candidates = list(lattice_points)
        candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
        ring2_candidates = candidates[:12]
        
    best_S = float('inf')
    best_config = None
    
    # Enumerate all 4-subsets of ring2_candidates
    # 12 choose 4 = 495
    for subset in itertools.combinations(range(12), 4):
        if time.time() > deadline - 5.0:
            break
            
        # Construct the 11 centers
        current_centers = [(0.0, 0.0)] + ring1 + [ring2_candidates[i] for i in subset]
        
        # Generate vertices for all 11 hexagons (angle 0)
        all_verts = []
        for cx, cy in current_centers:
            v = _hexagon_vertices(cx, cy, 0.0, 1.0)
            all_verts.append(v)
        
        vertices = np.vstack(all_verts)
        
        # Find min enclosing hexagon
        S, cx, cy, phi = _min_enclosing_hexagon(vertices)
        
        if S < best_S:
            best_S = S
            best_config = {
                'centers': current_centers,
                'cx': cx,
                'cy': cy,
                'S': S,
                'phi': phi
            }
            
    if best_config is None:
        # Fallback: just return the 7-hex flower, padded with 4 dummy ones far away? 
        # No, must be valid. 
        # If no valid config found (shouldn't happen), return a large hexagon.
        centers = [(0.0, 0.0)] + ring1 + ring2_candidates[:4]
        all_verts = []
        for cx, cy in centers:
            v = _hexagon_vertices(cx, cy, 0.0, 1.0)
            all_verts.append(v)
        vertices = np.vstack(all_verts)
        S, cx, cy, phi = _min_enclosing_hexagon(vertices)
        best_config = {
            'centers': centers,
            'cx': cx,
            'cy': cy,
            'S': S,
            'phi': phi
        }
        
    inner = [[c[0], c[1], 0.0] for c in best_config['centers']]
    return inner, (best_config['cx'], best_config['cy']), best_config['S'], best_config['phi']
