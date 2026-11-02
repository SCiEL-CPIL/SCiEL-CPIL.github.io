import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal tiling basis vectors (spacing 2 between centers for unit hexagons)
    # e1 = (1, 0), e2 = (0.5, sqrt(3)/2), scaled by 2
    e1_base = np.array([2.0, 0.0])
    e2_base = np.array([1.0, np.sqrt(3.0)])
    
    # 3x4 grid: i in {0,1,2}, j in {0,1,2,3}
    centers_base = []
    for i in range(3):
        for j in range(4):
            centers_base.append(i * e1_base + j * e2_base)
    centers_base = np.array(centers_base)
    
    # Apply affine shear: transform basis vectors
    # New e1 = e1_base + s * e2_base (shear along e2)
    # New e2 = e2_base (or also sheared)
    # Actually, let's use a general 2x2 affine transform on the basis
    # A = [[a, b], [c, d]] applied to [e1_base, e2_base]
    # To keep it low-dim, use shear: e1' = e1_base + s*e2_base, e2' = e2_base
    # Then rotate the whole thing by angle theta
    
    def compute_positions(s, theta):
        e1 = e1_base + s * e2_base
        e2 = e2_base
        centers = np.array([i * e1 + j * e2 for i in range(3) for j in range(4)])
        # Rotate
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        centers = centers @ R.T
        return centers
    
    def bounding_hexagon_side(centers, rotation):
        # For each unit hexagon at center c, its vertices are at c + R_k * (1,0) rotated by hex rotation
        # But we want the minimal regular hexagon (with some orientation) enclosing all hexagons.
        # A regular hexagon of side S centered at origin with orientation phi has vertices at
        # S * (cos(phi + k*pi/3), sin(phi + k*pi/3)) for k=0..5
        # The hexagon is the intersection of 6 half-planes.
        # For a point p to be inside: for each direction u_k (unit normal to each side),
        # p . u_k <= inradius = S * sqrt(3)/2
        # So S >= 2/sqrt(3) * max_k (p . u_k) for all p in all hexagons.
        # Each hexagon's max extent in direction u is |c . u| + 1 (circumradius)
        # Actually, for a unit hexagon, the max projection in any direction u is 1 (circumradius).
        # So for direction u_k, the max over all hexagons is max_c (c . u_k) + 1
        # And min is min_c (c . u_k) - 1
        # The width in direction u_k is max - min = max_c(c.u_k) - min_c(c.u_k) + 2
        # For a regular hexagon, the width in each of the 6 side-normal directions must be <= 2 * inradius = 2*S*sqrt(3)/2 = S*sqrt(3)
        # So S >= width_k / sqrt(3) for each k
        # S = max_k (width_k / sqrt(3))
        
        # Side normals of regular hexagon with orientation phi:
        # The sides are perpendicular to directions at angles phi + k*pi/3 + pi/6? 
        # Let me think: regular hexagon with a vertex at angle phi.
        # Vertices at phi + k*60 deg.
        # Side normals (outward) are at angles phi + 30 + k*60 deg.
        # Actually, for a regular hexagon centered at origin with "pointy top" (vertex at 90 deg),
        # the side normals are at 0, 60, 120, 180, 240, 300 deg.
        # Let's say the hexagon has orientation alpha, meaning one vertex is at angle alpha.
        # Then side normals are at alpha + 30 + k*60 deg for k=0..5.
        
        best_S = float('inf')
        # Try different orientations of the bounding hexagon
        # The optimal orientation should align with the packing, so search over alpha
        for alpha in np.linspace(0, np.pi/3, 100):
            # Side normal directions
            angles = alpha + np.pi/6 + np.arange(6) * np.pi/3
            S_for_alpha = 0
            for ang in angles:
                u = np.array([np.cos(ang), np.sin(ang)])
                projections = centers @ u
                width = projections.max() - projections.min() + 2.0  # +2 for circumradius of each hexagon
                S_candidate = width / np.sqrt(3.0)
                if S_candidate > S_for_alpha:
                    S_for_alpha = S_candidate
            if S_for_alpha < best_S:
                best_S = S_for_alpha
        return best_S
    
    def objective(params):
        s, theta = params
        centers = compute_positions(s, theta)
        # Center the packing at origin for symmetry
        centers = centers - centers.mean(axis=0)
        return bounding_hexagon_side(centers, 0.0)
    
    # Optimize shear s and rotation theta
    best_result = None
    best_val = float('inf')
    
    # Try multiple initial guesses
    for s0 in np.linspace(-1.0, 1.0, 11):
        for theta0 in np.linspace(0, np.pi/6, 11):
            res = minimize(objective, [s0, theta0], method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
    
    s_opt, theta_opt = best_result
    
    # Compute final positions
    centers = compute_positions(s_opt, theta_opt)
    centers = centers - centers.mean(axis=0)
    
    # Build inner hexagons
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
    
    # The bounding hexagon is centered at origin with side length best_val
    # Its orientation: we need to find the optimal alpha
    # For now, use the alpha that gave best_S in the last evaluation
    # Let's recompute to find optimal alpha
    best_alpha = 0.0
    best_S_final = float('inf')
    for alpha in np.linspace(0, np.pi/3, 200):
        angles = alpha + np.pi/6 + np.arange(6) * np.pi/3
        S_for_alpha = 0
        for ang in angles:
            u = np.array([np.cos(ang), np.sin(ang)])
            projections = centers @ u
            width = projections.max() - projections.min() + 2.0
            S_candidate = width / np.sqrt(3.0)
            if S_candidate > S_for_alpha:
                S_for_alpha = S_candidate
        if S_for_alpha < best_S_final:
            best_S_final = S_for_alpha
            best_alpha = alpha
    
    # Convert alpha to rotation angle in degrees for the outer hexagon
    # The outer hexagon's rotation: if a vertex is at angle best_alpha, 
    # and we define rotation as the angle of the first vertex from x-axis
    outer_rotation_deg = np.degrees(best_alpha)
    
    return inner, (0.0, 0.0), float(best_S_final), float(outer_rotation_deg)
# EVOLVE-BLOCK-END
