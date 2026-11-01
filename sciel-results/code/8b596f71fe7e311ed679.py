"""Pack 11 unit hexagons in smallest regular hexagon using fixed contact graph + 1D rotation optimization."""
import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

def _hex_support(vertices, directions):
    """Support function of a polygon given its vertices and query directions."""
    return np.max(vertices @ directions.T, axis=0)

def _hexagon_vertices(cx, cy, rotation_deg):
    """Vertices of unit hexagon centered at (cx,cy) with rotation."""
    a = np.radians(rotation_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _max_container_side(inner_positions, inner_rotations, outer_rotation_deg, n_dirs=720):
    """Find minimum container side for given inner positions and outer rotation, container at origin."""
    theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = np.column_stack([np.cos(theta), np.sin(theta)])
    
    beta = np.radians(outer_rotation_deg)
    cont_angles = beta + np.arange(6) * np.pi / 3
    cont_unit_verts = np.column_stack([np.cos(cont_angles), np.sin(cont_angles)])
    
    max_inner_support = np.zeros(n_dirs)
    for (px, py), rot in zip(inner_positions, inner_rotations):
        verts = _hexagon_vertices(px, py, rot)
        h = _hex_support(verts, dirs)
        max_inner_support = np.maximum(max_inner_support, h)
    
    cont_support_unit = _hex_support(cont_unit_verts, dirs)
    cont_support_unit = np.maximum(cont_support_unit, 1e-12)
    
    s_required = np.max(max_inner_support / cont_support_unit)
    return s_required

def _build_lattice_packing(inner_rotation_deg):
    """Build 11 hexagons in 3-4-3 triangular lattice."""
    d = np.sqrt(3)
    
    positions_axial = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
    ]
    
    positions = []
    for q, r in positions_axial:
        x = d * (q + r / 2.0)
        y = d * np.sqrt(3) / 2.0 * r
        positions.append((x, y))
    
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [(p[0] - cx, p[1] - cy) for p in positions]
    
    rotations = [inner_rotation_deg] * 11
    return positions, rotations

def _objective_2d(params):
    """Objective with separate inner and outer rotation."""
    inner_rot, outer_rot = params
    positions, rotations = _build_lattice_packing(inner_rot)
    return _max_container_side(positions, rotations, outer_rot, n_dirs=720)

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    # Coarse 2D grid search
    best_score = float('inf')
    best_inner = 0.0
    best_outer = 0.0
    
    n_coarse = 30
    for i in range(n_coarse):
        if time.time() - start_time > deadline:
            break
        inner_angle = i * 60.0 / n_coarse
        for j in range(n_coarse):
            outer_angle = j * 60.0 / n_coarse
            positions, rotations = _build_lattice_packing(inner_angle)
            s = _max_container_side(positions, rotations, outer_angle, n_dirs=360)
            if s < best_score:
                best_score = s
                best_inner = inner_angle
                best_outer = outer_angle
    
    # Refine
    if time.time() - start_time < deadline:
        from scipy.optimize import minimize
        res = minimize(_objective_2d, x0=[best_inner, best_outer], 
                      bounds=[(-5, 65), (-5, 65)], method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or res.fun < best_score:
            best_inner, best_outer = res.x
            best_score = res.fun
    
    # Final polish
    if time.time() - start_time < deadline:
        res2 = minimize(_objective_2d, x0=[best_inner, best_outer],
                       bounds=[(best_inner-0.5, best_inner+0.5), (best_outer-0.5, best_outer+0.5)],
                       method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-14})
        if res2.fun < best_score:
            best_inner, best_outer = res2.x
            best_score = res2.fun
    
    positions, rotations = _build_lattice_packing(best_inner)
    inner = [[pos[0], pos[1], rot] for pos, rot in zip(positions, rotations)]
    
    return inner, (0.0, 0.0), best_score, best_outer

# EVOLVE-BLOCK-START
# construct defined above
# EVOLVE-BLOCK-END
