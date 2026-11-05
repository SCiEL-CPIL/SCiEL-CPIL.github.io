"""Packing 12 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side length, rotation in degrees."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_side_length(inner_positions, inner_angles, outer_center, outer_rotation_deg):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    cx, cy = outer_center
    outer_rot = np.radians(outer_rotation_deg)
    
    # Face normals of outer hexagon (6 directions)
    normals = np.array([np.cos(outer_rot + np.arange(6) * np.pi / 3),
                        np.sin(outer_rot + np.arange(6) * np.pi / 3)]).T
    
    max_proj = 0.0
    for i in range(N):
        px, py = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(px, py, 1.0, ang)
        # Project all vertices relative to outer center
        rel = verts - np.array([cx, cy])
        projs = rel @ normals  # shape (6, 6)
        max_proj = max(max_proj, np.max(np.abs(projs)))
    
    # Apothem = s * sqrt(3)/2, so s = 2 * apothem / sqrt(3) = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def _objective(params):
    """Objective: outer side length. params = [x1,y1,a1, ..., x12,y12,a12, ox,oy, orot]"""
    positions = np.array(params[:3*N]).reshape(N, 3)
    ox, oy, orot = params[3*N], params[3*N+1], params[3*N+2]
    return _outer_side_length(positions[:, :2], positions[:, 2], (ox, oy), orot)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial configuration: hexagonal lattice pattern
    # Hex lattice: spacing = sqrt(3) for edge-to-edge
    spacing = np.sqrt(3.0)
    
    # Create a hexagonal cluster of 12 hexagons
    # Pattern: rows of hexagons
    # Let's use a symmetric arrangement
    # Center at origin, place hexagons in hex lattice positions
    
    # Hex lattice positions for a cluster
    # Using axial coordinates
    positions = []
    # A good pattern for 12: 
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Or a more compact hexagonal shape
    
    # Let's try: a hexagon of hexagons with some extra
    # Central + 6 around + 5 more
    # Or simply a 4x3 grid with hex offset
    
    # Better: use a hexagonal number arrangement
    # 1 + 6 + 5 = 12, or 3+4+3+2 = 12
    
    # Let's use a compact arrangement:
    # Positions in hex lattice (axial coords q,r)
    axial = [
        (0, 0),    # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1 (6)
        (2, 0), (2, -1), (0, -2), (-2, 0), (-2, 1)  # 5 more from ring 2
    ]
    
    # Convert axial to pixel coords
    for q, r in axial[:N]:
        x = spacing * (q + r/2.0)
        y = spacing * r * np.sqrt(3)/2.0
        positions.append([x, y, 0.0])
    
    positions = np.array(positions)
    
    # Center of mass
    com = np.mean(positions[:, :2], axis=0)
    positions[:, :2] -= com
    
    # Initial params
    init_params = np.zeros(3*N + 3)
    for i in range(N):
        init_params[3*i] = positions[i, 0]
        init_params[3*i+1] = positions[i, 1]
        init_params[3*i+2] = 0.0
    init_params[3*N] = 0.0    # outer center x
    init_params[3*N+1] = 0.0  # outer center y
    init_params[3*N+2] = 0.0  # outer rotation
    
    best_params = init_params.copy()
    best_val = _objective(init_params)
    
    # Try multiple outer rotations
    for trial_rot in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        p = init_params.copy()
        p[3*N+2] = trial_rot
        res = minimize(_objective, p, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() > deadline:
            break
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    
    # Additional local refinement with different initial perturbations
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        p = best_params.copy()
        # Small random perturbation
        p[:3*N] += rng.normal(0, 0.1, 3*N)
        res = minimize(_objective, p, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() > deadline:
            break
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        res = minimize(_objective, best_params, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    ox = float(best_params[3*N])
    oy = float(best_params[3*N+1])
    orot = float(best_params[3*N+2])
    side = float(best_val)
    
    return inner, (ox, oy), side, orot

