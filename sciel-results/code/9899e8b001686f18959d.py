# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon."""
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _min_enclosing_side(positions):
    """
    Given 11 unit hexagons at angle 0°, find the minimum side length of a regular
    hexagon (centered at origin, angle=0) that contains all inner hex vertices.
    positions: (11, 2) array of centers.
    """
    # For a regular hexagon with side s at angle 0, the 6 edge normals are at 30, 90, 150, 210, 270, 330 degrees
    normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))],
    ])
    
    apothem_factor = np.sqrt(3) / 2
    
    max_proj = 0.0
    for i in range(11):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / apothem_factor

def _check_separation(positions, tol=1e-6):
    """Check that all pairs of unit hexagons (angle 0) are separated."""
    n = 11
    # Precompute vertices for all hexagons
    all_verts = []
    for i in range(n):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    
    # 12 edge normals for hexagons at angle 0: 6 from hex i, 6 from hex j (same set since both at angle 0)
    # Actually, for two hexagons both at angle 0, the 12 normals are just the 6 unique normals (each appears twice)
    # But to be safe, we check all 12 as per the spec
    normals_i = []
    normals_j = []
    for k in range(6):
        a = np.radians(0) + np.radians(30 + 60 * k)
        normals_i.append([np.cos(a), np.sin(a)])
        normals_j.append([np.cos(a), np.sin(a)])
    all_normals = np.array(normals_i + normals_j)
    
    for i in range(n):
        for j in range(i+1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            separated = False
            for normal in all_normals:
                ni = vi @ normal
                nj = vj @ normal
                if max(ni) < min(nj) - tol or max(nj) < min(ni) - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    return True

def _check_containment(positions, outer_side, tol=-1e-6):
    """Check all inner vertices are inside the outer hexagon (centered at origin, angle 0)."""
    outer_verts = _hex_vertices(0.0, 0.0, outer_side, 0.0)
    for i in range(11):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        for p in verts:
            for i_edge in range(6):
                p1 = outer_verts[i_edge]
                p2 = outer_verts[(i_edge+1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < tol:
                    return False
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    d = np.sqrt(3)
    
    # Initial arrangement: center + 6 surrounding + 4 more, all at angle 0
    initial_positions = []
    initial_positions.append([0.0, 0.0])
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        initial_positions.append([d * np.cos(angle), d * np.sin(angle)])
    # 4 more: place at 2*d in directions 30, 150, 270, 30 (skip 90, 210 to avoid overlap)
    for k in [0, 2, 4, 1]:
        angle = np.radians(30 + 60 * k)
        initial_positions.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    
    initial_positions = np.array(initial_positions)
    
    def objective(params):
        """Minimize the enclosing hexagon side length."""
        positions = params.reshape(11, 2)
        return _min_enclosing_side(positions)
    
    def _is_valid(params):
        positions = params.reshape(11, 2)
        if not _check_separation(positions):
            return False
        side = _min_enclosing_side(positions)
        if not _check_containment(positions, side):
            return False
        return True
    
    initial_side = _min_enclosing_side(initial_positions)
    initial_valid = _check_separation(initial_positions) and _check_containment(initial_positions, initial_side)
    
    best_positions = initial_positions.copy()
    best_side = initial_side
    
    if not initial_valid:
        # Scale up to make valid
        scale = 1.1
        while not _check_separation(initial_positions) and scale < 3.0:
            initial_positions = initial_positions * scale
            scale += 0.05
        best_positions = initial_positions.copy()
        best_side = _min_enclosing_side(best_positions)
    
    # SLSQP optimization with symmetry-reduced DOF (22 instead of 33)
    if time.time() < deadline - 120.0:
        try:
            bounds = [(-8, 8)] * 22
            res = minimize(
                objective,
                best_positions.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'iprint': 0}
            )
            if res.fun < best_side and _is_valid(res.x) and time.time() < deadline - 60.0:
                best_positions = res.x.reshape(11, 2)
                best_side = res.fun
        except Exception:
            pass
    
    # Multiple restarts with perturbations
    if time.time() < deadline - 60.0:
        np.random.seed(123)
        for restart in range(50):
            if time.time() > deadline - 30.0:
                break
            
            perturbation = np.random.randn(22) * 0.5
            trial_params = best_positions.flatten() + perturbation
            
            if not _is_valid(trial_params):
                continue
            
            trial_side = objective(trial_params)
            if trial_side < best_side:
                best_positions = trial_params.reshape(11, 2)
                best_side = trial_side
    
    # Final local polish
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-8, 8)] * 22
            res = minimize(
                objective,
                best_positions.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'iprint': 0}
            )
            if res.fun < best_side and _is_valid(res.x):
                best_positions = res.x.reshape(11, 2)
                best_side = res.fun
        except Exception:
            pass
    
    # Construct inner array: (11, 3) with (x, y, angle_deg)
    inner = np.column_stack([best_positions, np.zeros(11)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(best_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
