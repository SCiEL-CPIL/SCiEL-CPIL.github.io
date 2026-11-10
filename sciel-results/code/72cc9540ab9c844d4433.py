# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(a_deg):
    """Get 6 outward edge normals for a hexagon with given angle."""
    a = np.deg2rad(a_deg)
    # Edge normals are perpendicular to edges, pointing outward
    # For a hexagon with vertices at angles a + 60i, the edge between vertex i and i+1
    # has normal at angle a + 60*i + 30
    angles = a + np.deg2rad(60 * np.arange(6) + 30)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _separation_gap(vi, vj, ni, nj):
    """Check if two hexagons are separated. Returns min gap (positive = separated)."""
    # ni, nj are (6,2) arrays of outward normals
    # For each normal, compute max projection on hex i and min on hex j
    # If max_i < min_j - eps, they're separated in that direction
    # We need to check all 12 normals (6 from each)
    min_gap = np.inf
    for n in ni:
        proj_i = vi @ n
        proj_j = vj @ n
        gap = np.min(proj_j) - np.max(proj_i)
        min_gap = min(min_gap, gap)
    for n in nj:
        proj_i = vi @ n
        proj_j = vj @ n
        gap = np.min(proj_i) - np.max(proj_j)
        min_gap = min(min_gap, gap)
    return min_gap

def _min_enclosing_hex_side(inner_data, outer_angle_deg):
    """Find minimum side length of outer hexagon (centered at origin) with given angle."""
    a_outer = np.deg2rad(outer_angle_deg)
    # 6 inward edge normals of outer hexagon
    normals_angles = a_outer + np.deg2rad(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    # Get all vertices of all inner hexagons
    all_pts = []
    for i in range(len(inner_data)):
        x, y, ang = inner_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    cos30 = np.cos(np.deg2rad(30))
    max_proj = 0
    for n in normals:
        proj = all_pts @ n
        max_proj = max(max_proj, np.max(proj))
    
    return max_proj / cos30

def _soft_max(x, temperature=0.01):
    """Soft max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + temperature * np.log(np.sum(np.exp((x - x_max) / temperature)))

def _objective(params, use_soft=True):
    """Objective: minimize outer hex side length with separation constraints."""
    inner = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    # Compute outer side length
    a_outer = np.deg2rad(outer_angle)
    normals_angles = a_outer + np.deg2rad(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    # Get all vertices
    all_pts = []
    hex_verts = []
    hex_normals = []
    for i in range(11):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_pts.append(verts)
        hex_verts.append(verts)
        hex_normals.append(_edge_normals(ang))
    all_pts = np.vstack(all_pts)
    
    # Compute projections for outer containment
    cos30 = np.cos(np.deg2rad(30))
    projections = []
    for n in normals:
        proj = all_pts @ n
        projections.append(np.max(proj))
    projections = np.array(projections)
    
    if use_soft:
        side = _soft_max(projections, 0.01) / cos30
    else:
        side = np.max(projections) / cos30
    
    # Check separation between all pairs
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            gap = _separation_gap(hex_verts[i], hex_verts[j], hex_normals[i], hex_normals[j])
            if gap < 1e-6:
                # Overlapping or too close
                penalty += max(0, 1e-6 - gap) ** 2
    
    return side + 1000 * penalty

def _initial_guess():
    """Create initial hexagonal cluster arrangement."""
    apothem = np.sqrt(3) / 2
    center_dist = 2 * apothem  # = sqrt(3)
    
    # Hexagonal lattice positions for 11 hexagons
    # Using axial coordinates
    positions = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (2, 0),
        (0, 2),
    ]
    
    inner = np.zeros((11, 3))
    for i, (q, r) in enumerate(positions):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        inner[i] = [x, y, 0]
    
    outer_angle = 0.0
    params = np.concatenate([inner.flatten(), [outer_angle]])
    return params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    # Get initial guess
    params0 = _initial_guess()
    
    # First, do a quick optimization with soft objective
    best_params = params0.copy()
    best_obj = _objective(params0, use_soft=True)
    
    # Try L-BFGS-B with soft objective
    try:
        result = minimize(
            _objective,
            params0,
            args=(True,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    # Try multiple restarts with different initial conditions
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Different initial angles and perturbations
        np.random.seed(100 + attempt)
        
        # Try different outer angles
        outer_angle_try = np.random.uniform(0, 60)
        
        # Perturb positions
        params_try = params0.copy()
        params_try[33] = outer_angle_try
        params_try[:33] += np.random.normal(0, 0.05, size=33)
        
        try:
            result = minimize(
                _objective,
                params_try,
                args=(True,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1500}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final polish with exact (non-soft) objective
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(False,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
            )
            if result.fun < _objective(best_params, use_soft=False):
                best_params = result.x
        except Exception:
            pass
    
    # Extract best solution
    inner = best_params[:33].reshape(11, 3)
    outer_angle = best_params[33]
    
    # Compute the actual outer side length
    outer_side = _min_enclosing_hex_side(inner, outer_angle)
    
    # The outer hexagon is centered at origin
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    # Verify validity
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_hex_data)):
        # Fallback
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    if outer_side < np.sqrt(11):
        outer_side = np.sqrt(11)
    
    return inner, outer_hex_data, outer_side

# EVOLVE-BLOCK-END
