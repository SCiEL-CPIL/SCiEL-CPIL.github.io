# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _get_all_inner_vertices(positions):
    """Return (72, 2) array of all vertices of 12 unit hexagons at given positions (angle 0)."""
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    for i in range(n):
        all_verts[i*6:(i+1)*6] = positions[i] + base_verts
    return all_verts

def _check_separation_fast(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Returns True if separated, False otherwise.
    Uses Separating Axis Theorem with vectorized computation.
    """
    n = len(positions)
    # Precompute vertices for all hexagons: (n, 6, 2)
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    
    # Precompute edge normals for a unit hexagon at angle 0
    edges = np.empty((6, 2))
    for k in range(6):
        edges[k] = base_verts[(k+1) % 6] - base_verts[k]
    normals = np.empty((6, 2))
    for k in range(6):
        normals[k] = np.array([-edges[k][1], edges[k][0]])
    
    # Precompute projections of base vertices onto each of the 6 normals
    # base_dots[v, k] = dot(base_verts[v], normals[k])
    base_dots = base_verts @ normals.T  # (6, 6)
    
    # center_dots[i, k] = dot(positions[i], normals[k])
    center_dots = positions @ normals.T  # (n, 6)
    
    # min/max projections of base hexagon onto each normal
    base_min = base_dots.min(axis=0)  # (6,)
    base_max = base_dots.max(axis=0)  # (6,)
    
    for i in range(n):
        for j in range(i+1, n):
            # Projection intervals for hex i and j onto each of 6 normals
            proj_i_min = center_dots[i] + base_min  # (6,)
            proj_i_max = center_dots[i] + base_max  # (6,)
            proj_j_min = center_dots[j] + base_min  # (6,)
            proj_j_max = center_dots[j] + base_max  # (6,)
            
            # Check if any normal separates them
            # Separation: proj_i_max[k] < proj_j_min[k] - eps OR proj_j_max[k] < proj_i_min[k] - eps
            sep1 = proj_i_max < proj_j_min - 1e-6
            sep2 = proj_j_max < proj_i_min - 1e-6
            if np.any(sep1) or np.any(sep2):
                continue
            else:
                return False
    return True

def _compute_s(positions, angle_deg, outer_cx, outer_cy):
    """Compute the minimum outer hexagon side length to contain all inner hexagons."""
    all_verts = _get_all_inner_vertices(positions)
    shifted = all_verts - np.array([outer_cx, outer_cy])
    
    a_rad = np.radians(angle_deg)
    # Normals are at angle theta + 30 + 60k
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = shifted @ normals.T
    max_dot = np.max(dots)
    
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # The maximum projection onto a normal direction equals the apothem
    # So s*sqrt(3)/2 = max_dot => s = 2*max_dot/sqrt(3)
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _objective(params):
    """
    Objective function for optimization.
    params: (27,) array. First 24 are (x1, y1, ..., x12, y12). 24th is angle_deg. 25,26 are outer center (cx, cy).
    Returns s (outer hexagon side length) or penalty.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    outer_cx = params[25]
    outer_cy = params[26]
    
    # Check separation
    if not _check_separation_fast(positions):
        return 1000.0  # Large penalty
    
    return _compute_s(positions, angle_deg, outer_cx, outer_cy)

def _hex_lattice_positions(spacing_scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    s = spacing_scale
    h = 2.5 * s
    v = 2.165 * s  # sqrt(3)/2 * 2.5
    
    positions = np.array([
        [0.0, 0.0],
        [-h, 0.0],
        [h, 0.0],
        [-h/2, v],
        [h/2, v],
        [-h/2, -v],
        [h/2, -v],
        [-3*h/2, v],
        [3*h/2, v],
        [-3*h/2, -v],
        [3*h/2, -v],
        [0.0, -2*v],
    ])
    return positions

def _circular_positions(radius=2.0):
    """Generate positions on a circle."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return positions

def _compute_optimal_outer_center(positions, angle_deg):
    """Compute the optimal outer center that minimizes the containing hexagon size."""
    all_verts = _get_all_inner_vertices(positions)
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # The optimal center minimizes max_k (max_v dot(vert_v - center, normal_k))
    # This is a convex optimization problem. For a hexagon, the optimal center
    # is the one that equalizes the maximum projections on opposite normals.
    # For simplicity, use the centroid of all vertices as a good approximation.
    center = np.mean(all_verts, axis=0)
    return center

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    # Initial configurations
    inits = []
    
    # Hexagonal lattice
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    
    # Scaled lattices
    for scale in [0.9, 1.1]:
        inits.append(_hex_lattice_positions(scale))
        
    # Circular
    for r in [1.5, 2.0, 2.5]:
        inits.append(_circular_positions(r))
        
    # Random
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 3.0)
        
    # Perturbed lattice
    for _ in range(5):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.5)

    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Compute optimal outer center for this configuration
        init_center = _compute_optimal_outer_center(init_pos, 0.0)
        
        # Try a few initial angles for the outer hexagon
        for init_angle in [0, 30, 10, -10]:
            if time.time() > deadline - 20:
                break
                
            # Recompute optimal center for this angle
            init_center = _compute_optimal_outer_center(init_pos, init_angle)
            
            # Construct initial params
            init_params = np.zeros(27)
            init_params[:24] = init_pos.flatten()
            init_params[24] = init_angle
            init_params[25] = init_center[0]
            init_params[26] = init_center[1]
            
            # Run Nelder-Mead
            try:
                res = minimize(
                    _objective, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                continue
                
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
                
    # If optimization failed or didn't find a valid solution, fallback to lattice
    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions(1.0)
        best_angle = 0
        best_center = _compute_optimal_outer_center(best_pos, best_angle)
        best_s = _compute_s(best_pos, best_angle, best_center[0], best_center[1])
        best_params = np.zeros(27)
        best_params[:24] = best_pos.flatten()
        best_params[24] = best_angle
        best_params[25] = best_center[0]
        best_params[26] = best_center[1]

    # Final high-precision polish pass
    if time.time() < deadline - 10:
        try:
            res_polish = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Extract results
    final_positions = best_params[:24].reshape(12, 2)
    final_angle = best_params[24]
    final_outer_cx = best_params[25]
    final_outer_cy = best_params[26]
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([final_outer_cx, final_outer_cy, final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
