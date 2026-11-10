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
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    
    edges = np.empty((6, 2))
    for k in range(6):
        edges[k] = base_verts[(k+1) % 6] - base_verts[k]
    normals = np.empty((6, 2))
    for k in range(6):
        normals[k] = np.array([-edges[k][1], edges[k][0]])
    
    base_dots = base_verts @ normals.T  # (6, 6)
    center_dots = positions @ normals.T  # (n, 6)
    
    base_min = base_dots.min(axis=0)  # (6,)
    base_max = base_dots.max(axis=0)  # (6,)
    
    for i in range(n):
        for j in range(i+1, n):
            proj_i_min = center_dots[i] + base_min
            proj_i_max = center_dots[i] + base_max
            proj_j_min = center_dots[j] + base_min
            proj_j_max = center_dots[j] + base_max
            
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
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = shifted @ normals.T
    max_dot = np.max(dots)
    
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _min_containing_hex_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with given angle
    that contains all vertices of the inner hexagons, optimizing the center.
    """
    all_verts = _get_all_inner_vertices(positions)
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Initial guess for center: centroid of vertices
    center_init = np.mean(all_verts, axis=0)
    
    def center_obj(center):
        shifted = all_verts - center
        dots = shifted @ normals.T
        max_dot = np.max(dots)
        s = (2.0 / np.sqrt(3.0)) * max_dot
        return s

    res = minimize(center_obj, center_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    return res.fun, res.x

def _objective_full(params):
    """
    Full objective with separation penalty.
    params: (25,) array. First 24 are (x1, y1, ..., x12, y12). 24th is angle_deg.
    Outer center is optimized numerically.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    if not _check_separation_fast(positions):
        return 1000.0
    
    # Optimize outer center for given positions and angle
    s, center = _min_containing_hex_side(positions, angle_deg)
    
    # Check if s is too small (invalid packing)
    if s < np.sqrt(12) - 1e-6:
        return 1000.0
        
    return s

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

def _ring_positions():
    """Generate a ring of 6 around a center of 6."""
    h = 2.5
    v = 2.165
    positions = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [-h, 0.0],
        [h/2, v],
        [-h/2, v],
        [h/2, -v],
        [-h/2, -v],
        [2*h, 0.0],
        [-2*h, 0.0],
        [3*h/2, v],
        [-3*h/2, v],
        [3*h/2, -v],
    ])
    return positions

def _compact_positions():
    """Generate a more compact 3-4-3-2 arrangement."""
    h = 2.5
    v = 2.165
    positions = np.array([
        # Row 1: 3 hexagons
        [-h, 0.0], [0.0, 0.0], [h, 0.0],
        # Row 2: 4 hexagons
        [-1.5*h, v], [-0.5*h, v], [0.5*h, v], [1.5*h, v],
        # Row 3: 3 hexagons
        [-h, 2*v], [0.0, 2*v], [h, 2*v],
        # Row 4: 2 hexagons
        [-0.5*h, 3*v], [0.5*h, 3*v],
    ])
    return positions

def _optimized_lattice():
    """Generate an optimized lattice-like arrangement."""
    h = 2.5
    v = 2.165
    # 4-4-4 arrangement
    positions = np.array([
        [-1.5*h, 0.0], [-0.5*h, 0.0], [0.5*h, 0.0], [1.5*h, 0.0],
        [-2.0*h, v], [-1.0*h, v], [0.0, v], [1.0*h, v],
        [-1.5*h, 2*v], [-0.5*h, 2*v], [0.5*h, 2*v], [1.5*h, 2*v],
    ])
    return positions

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
        
    # Ring
    inits.append(_ring_positions())
    
    # Compact
    inits.append(_compact_positions())
    
    # Optimized lattice
    inits.append(_optimized_lattice())
    
    # Random
    for _ in range(8):
        inits.append(np.random.randn(12, 2) * 3.0)
        
    # Perturbed lattice
    for _ in range(8):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.5)
    
    # Perturbed compact
    for _ in range(4):
        inits.append(_compact_positions() + np.random.randn(12, 2) * 0.3)
    
    # Perturbed ring
    for _ in range(4):
        inits.append(_ring_positions() + np.random.randn(12, 2) * 0.3)

    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        # Try a few initial angles for the outer hexagon
        for init_angle in [0, 30, 15, -15, 45]:
            if time.time() > deadline - 30:
                break
                
            # Construct initial params
            init_params = np.zeros(25)
            init_params[:24] = init_pos.flatten()
            init_params[24] = init_angle
            
            # Run Nelder-Mead with more iterations
            try:
                res = minimize(
                    _objective_full, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                continue
                
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
                
                # Check if we're close to optimal
                if best_s < 5.0:
                    break
    
    # If optimization failed or didn't find a valid solution, fallback to lattice
    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions(1.0)
        best_angle = 0
        all_verts = _get_all_inner_vertices(best_pos)
        best_center = np.mean(all_verts, axis=0)
        best_s = _compute_s(best_pos, best_angle, best_center[0], best_center[1])
        best_params = np.zeros(25)
        best_params[:24] = best_pos.flatten()
        best_params[24] = best_angle

    # Final high-precision polish pass
    if time.time() < deadline - 20:
        try:
            res_polish = minimize(
                _objective_full, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Second polish if time allows
    if time.time() < deadline - 10 and best_params is not None:
        try:
            res_polish2 = minimize(
                _objective_full, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res_polish2.fun < best_s:
                best_s = res_polish2.fun
                best_params = res_polish2.x.copy()
        except Exception:
            pass

    # Extract results
    final_positions = best_params[:24].reshape(12, 2)
    final_angle = best_params[24]
    
    # Compute optimal outer center
    final_s, center = _min_containing_hex_side(final_positions, final_angle)
    final_outer_cx, final_outer_cy = center
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([final_outer_cx, final_outer_cy, final_angle])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
