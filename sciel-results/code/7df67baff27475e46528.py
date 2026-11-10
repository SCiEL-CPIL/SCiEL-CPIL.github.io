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
    for i in range(n):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0)
        all_verts[i*6:(i+1)*6] = verts
    return all_verts

def _check_separation_fast(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Returns True if separated, False otherwise.
    Uses Separating Axis Theorem with vectorized computation.
    """
    n = len(positions)
    # Precompute vertices for all hexagons: (n, 6, 2)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0)
    
    # Precompute edge normals for a unit hexagon at angle 0
    # Edge k goes from vertex k to vertex (k+1)%6
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    edges = np.empty((6, 2))
    for k in range(6):
        edges[k] = base_verts[(k+1) % 6] - base_verts[k]
    normals = np.empty((6, 2))
    for k in range(6):
        e = edges[k]
        normals[k] = np.array([-e[1], e[0]])
    
    # For each pair (i, j), check all 12 separating axes (6 from i, 6 from j)
    for i in range(n):
        vi = verts[i]
        for j in range(i+1, n):
            vj = verts[j]
            # Check normals from hexagon i
            # Project both hexagons onto each normal
            for k in range(6):
                nrm = normals[k]
                proj_i = vi @ nrm
                proj_j = vj @ nrm
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    break  # Separated on this axis
            else:
                # Not separated on any of i's normals, check j's normals
                found = False
                for k in range(6):
                    nrm = normals[k]
                    proj_i = vi @ nrm
                    proj_j = vj @ nrm
                    if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                        found = True
                        break
                if not found:
                    return False
    return True

def _min_containing_hex_side(positions, angle_deg, center):
    """
    Compute the minimum side length of a regular hexagon with given angle and center
    that contains all vertices of the unit hexagons at the given positions.
    """
    all_verts = _get_all_inner_vertices(positions)
    # Shift vertices so outer center is at origin
    all_verts = all_verts - center
    
    # Outer hexagon normals: for a hexagon with angle a, the outward normals of the edges
    # are at angles a + 30 + 60k for k=0..5
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each normal direction, find the maximum projection
    dots = all_verts @ normals.T
    max_dots = np.max(dots, axis=0)
    
    # The half-width in each normal direction must be at least the max projection
    # For a regular hexagon with side s, the apothem (half-width) is s * sqrt(3)/2
    # So s = 2 * apothem / sqrt(3)
    # The required s is the maximum over all 6 directions of (2 * max_dot_k) / sqrt(3)
    s = (2.0 / np.sqrt(3.0)) * np.max(max_dots)
    
    return s

def _objective(params):
    """
    Objective function for optimization.
    params: (27,) array. First 24 are (x1, y1, ..., x12, y12). 
    Next 2 are outer center (cx, cy). Last is angle_deg.
    Returns s (outer hexagon side length) or penalty.
    """
    positions = params[:24].reshape(12, 2)
    center = params[24:26]
    angle_deg = params[26]
    
    # Check separation
    if not _check_separation_fast(positions):
        return 1000.0  # Large penalty
    
    # Compute min containing hexagon side
    s = _min_containing_hex_side(positions, angle_deg, center)
    
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

def _tight_ring_positions():
    """Generate a tight ring of 12 hexagons around a central one, with the 12th replacing center."""
    # 7 hexagons in a flower pattern, then add 5 more in a ring
    h = 2.5
    v = 2.165
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
    
    # Tight ring
    inits.append(_tight_ring_positions())
    
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Try a few initial angles for the outer hexagon
        for init_angle in [0, 30, 10, -10, 60, -30]:
            if time.time() > deadline - 20:
                break
                
            # Construct initial params
            init_params = np.zeros(27)
            init_params[:24] = init_pos.flatten()
            init_params[24:26] = [0.0, 0.0]
            init_params[26] = init_angle
            
            # Run Nelder-Mead
            try:
                res = minimize(
                    _objective, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
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
        best_center = np.array([0.0, 0.0])
        # Calculate s for this fallback
        best_s = _min_containing_hex_side(best_pos, best_angle, best_center)
        best_params = np.zeros(27)
        best_params[:24] = best_pos.flatten()
        best_params[24:26] = best_center
        best_params[26] = best_angle

    # Extract results
    final_positions = best_params[:24].reshape(12, 2)
    final_center = best_params[24:26]
    final_angle = best_params[26]
    
    # Recompute s accurately
    final_s = _min_containing_hex_side(final_positions, final_angle, final_center)
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([final_center[0], final_center[1], final_angle])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
