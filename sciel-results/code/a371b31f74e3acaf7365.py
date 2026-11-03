"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Place 11 unit hexagons in a compact cluster
    # Triangular lattice: spacing between adjacent hexagon centers = sqrt(3) for side-1 hexagons
    # Actually for unit hexagons (side 1), the distance between centers of adjacent hexagons in a tiling is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # Wait: inradius of unit hexagon = sqrt(3)/2. Adjacent hexagons in tiling have center distance = 2 * inradius = sqrt(3).
    
    # Let's use a layout: 
    # Row 0 (y=0): 4 hexagons at x = -1.5*d, -0.5*d, 0.5*d, 1.5*d where d = sqrt(3)
    # Row 1 (y=h): 3 hexagons offset
    # Row 2 (y=2h): 3 hexagons
    # Row 3 (y=3h): 1 hexagon
    # h = sqrt(3)/2 * ... let me think.
    
    # For hexagonal tiling with unit hexagons:
    # Center-to-center distance horizontally: 2 * inradius * 2 / sqrt(3)? No.
    # Inradius r = sqrt(3)/2. 
    # Horizontal spacing between centers in same row: 2*r = sqrt(3)
    # Vertical spacing between rows: 3/2 * side * ... 
    # Actually: the distance from center to vertex is 1 (circumradius).
    # For hexagons tiled edge-to-edge: center distance = 2 * inradius = sqrt(3)
    # Row offset: x-shift = sqrt(3)/2, y-shift = 3/2 * ... 
    # Let me just use: dx = sqrt(3), dy = 1.5, x_offset = sqrt(3)/2
    
    dx = np.sqrt(3)
    dy = 1.5
    xo = np.sqrt(3) / 2
    
    # Layout: 4-3-3-1 or 3-4-3-1
    # Let's try: row0: 3, row1: 4, row2: 3, row3: 1
    positions = []
    # Row 0: y=0, 3 hexagons
    for k in range(3):
        positions.append([dx * (k - 1), 0.0])
    # Row 1: y=dy, 4 hexagons, offset by xo
    for k in range(4):
        positions.append([dx * (k - 1.5) + xo, dy])
    # Row 2: y=2*dy, 3 hexagons
    for k in range(3):
        positions.append([dx * (k - 1), 2 * dy])
    # Row 3: y=3*dy, 1 hexagon
    positions.append([0.0, 3 * dy])
    
    positions = np.array(positions)
    
    # All inner hexagons are axis-aligned (angle 0)
    angles = np.zeros(N)
    
    # Compute all vertices of all inner hexagons
    def get_vertices(cx, cy, angle_deg):
        """Get 6 vertices of unit hexagon centered at (cx,cy) with rotation."""
        a = np.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + np.pi / 3 * i
            verts.append([cx + np.cos(theta), cy + np.sin(theta)])
        return np.array(verts)
    
    all_verts = []
    for i in range(N):
        all_verts.append(get_vertices(positions[i, 0], positions[i, 1], angles[i]))
    all_verts = np.vstack(all_verts)  # shape (6N, 2)
    
    # For a regular hexagon with center (ox, oy), side S, rotation phi:
    # Edge normals are at angles phi + 30° + 60°*k for k=0..5
    # Constraint: n_j . (v - center) <= S * sqrt(3)/2 for all vertices v, all j
    # So S >= (2/sqrt(3)) * max_j (n_j . (v - center))
    
    def outer_side(params):
        ox, oy, phi = params
        phi_rad = np.radians(phi)
        # Edge normals
        normals = []
        for k in range(6):
            theta = phi_rad + np.pi / 6 + np.pi / 3 * k
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)  # (6, 2)
        
        # For each vertex, compute max over normals of n . (v - center)
        rel = all_verts - np.array([ox, oy])  # (6N, 2)
        # proj[j, i] = normals[j] . rel[i]
        proj = rel @ normals.T  # (6N, 6)
        max_proj = np.max(proj, axis=0)  # (6,)
        max_val = np.max(max_proj)
        S = 2.0 / np.sqrt(3) * max_val
        return S
    
    # Optimize (ox, oy, phi)
    def objective(params):
        return outer_side(params)
    
    # Try multiple initial guesses
    best_S = float('inf')
    best_params = None
    
    # Center of mass of inner hexagons
    cx0 = np.mean(positions[:, 0])
    cy0 = np.mean(positions[:, 1])
    
    initial_guesses = [
        [cx0, cy0, 0.0],
        [cx0, cy0, 30.0],
        [cx0, cy0, 60.0],
        [cx0, cy0, 15.0],
        [cx0, cy0, 45.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except:
            pass
    
    if best_params is None:
        best_params = [cx0, cy0, 0.0]
        best_S = outer_side(best_params)
    
    # Refine with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x
        except:
            pass
    
    ox, oy, phi = best_params
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], angles[i]])
    
    return inner, (ox, oy), best_S, phi

# EVOLVE-BLOCK-END
