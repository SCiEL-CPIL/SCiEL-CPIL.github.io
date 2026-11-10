# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _get_all_inner_vertices(positions, angle=0.0):
    """Return (72, 2) array of all vertices of 12 unit hexagons at given positions."""
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        all_verts[i*6:(i+1)*6] = verts
    return all_verts

def _check_separation(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Uses Separating Axis Theorem. Returns True if all separated.
    """
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0))
    
    for i in range(n):
        vi = verts_list[i]
        for j in range(i+1, n):
            vj = verts_list[j]
            found_sep = False
            # Check edges of hexagon i
            for k in range(6):
                p1 = vi[k]
                p2 = vi[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    found_sep = True
                    break
            if found_sep:
                continue
            # Check edges of hexagon j
            for k in range(6):
                p1 = vj[k]
                p2 = vj[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    found_sep = True
                    break
            if not found_sep:
                return False
    return True

def _min_containing_hex_side(positions, angle_deg, outer_center):
    """
    Compute the minimum side length of a regular hexagon with given angle and center
    that contains all inner hexagons.
    """
    all_verts = _get_all_inner_vertices(positions)
    shifted = all_verts - outer_center
    
    a_rad = np.radians(angle_deg)
    # The 6 edge normals of a hexagon with angle a are at a + 30 + 60*k degrees
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = shifted @ normals.T  # (72, 6)
    max_dot = np.max(np.abs(dots))
    
    # For a hexagon with side s, the apothem is s*sqrt(3)/2
    # max_dot = apothem = s * sqrt(3) / 2
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _soft_max(x, beta=100.0):
    """Soft maximum using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp((x - x_max) * beta)) / len(x)) / beta

def _objective(params, use_soft=False, beta=50.0):
    """
    Objective: minimize outer hexagon side length s.
    params: (25,) array. First 24 are (x1, y1, ..., x12, y12). Last is angle_deg.
    Outer center is fixed at origin.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    # Check separation - if violated, return large penalty
    if not _check_separation(positions):
        return 1000.0
    
    outer_center = np.array([0.0, 0.0])
    s = _min_containing_hex_side(positions, angle_deg, outer_center)
    return s

def _objective_smooth(params, beta=50.0):
    """
    Smooth objective using soft constraints for separation.
    This is used for gradient-based optimization.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    # Compute separation violations (should be negative for valid packing)
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0))
    
    max_violation = 0.0
    for i in range(n):
        vi = verts_list[i]
        for j in range(i+1, n):
            vj = verts_list[j]
            min_gap = float('inf')
            for k in range(6):
                p1 = vi[k]
                p2 = vi[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                gap1 = proj_i.max() - proj_j.min()
                gap2 = proj_j.max() - proj_i.min()
                min_gap = min(min_gap, max(gap1, gap2))
            for k in range(6):
                p1 = vj[k]
                p2 = vj[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                gap1 = proj_i.max() - proj_j.min()
                gap2 = proj_j.max() - proj_i.min()
                min_gap = min(min_gap, max(gap1, gap2))
            # min_gap should be <= 0 for non-overlapping (with margin)
            if min_gap > 0:
                max_violation = max(max_violation, min_gap)
    
    if max_violation > 1e-6:
        return 1000.0 + max_violation * 10
    
    outer_center = np.array([0.0, 0.0])
    s = _min_containing_hex_side(positions, angle_deg, outer_center)
    return s

def _hex_lattice_positions():
    """
    Generate a hexagonal lattice arrangement of 12 unit hexagons.
    Center + 6 neighbors + 5 more in a compact arrangement.
    """
    # For unit hexagons, adjacent centers are 2*apothem = 2*(sqrt(3)/2) = sqrt(3) apart
    # Actually for touching hexagons, center distance = 2 * apothem = sqrt(3)
    # Wait, for regular hexagon with side 1, apothem = sqrt(3)/2, so touching distance = sqrt(3)
    
    d = np.sqrt(3)  # center-to-center distance for touching hexagons
    
    positions = np.array([
        [0.0, 0.0],                    # center
        [d, 0.0],                      # right
        [-d, 0.0],                     # left
        [d/2, d*np.sqrt(3)/2],         # upper right
        [-d/2, d*np.sqrt(3)/2],        # upper left
        [d/2, -d*np.sqrt(3)/2],        # lower right
        [-d/2, -d*np.sqrt(3)/2],       # lower left
        [0.0, d*np.sqrt(3)],           # top
        [0.0, -d*np.sqrt(3)],          # bottom
        [d*1.5, d*np.sqrt(3)/2],       # far upper right
        [-d*1.5, d*np.sqrt(3)/2],      # far upper left
        [d*1.5, -d*np.sqrt(3)/2],      # far lower right
    ])
    return positions

def _compact_12_positions():
    """
    A more compact arrangement: center, 6 ring, and 5 more placed in gaps.
    """
    d = np.sqrt(3)
    positions = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [-d, 0.0],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        # Additional 5 - place in the "gaps" between ring hexagons
        [0.0, d*np.sqrt(3)*0.866],
        [d*0.866, d*np.sqrt(3)*0.866*0.5],
        [-d*0.866, d*np.sqrt(3)*0.866*0.5],
        [d*0.866, -d*np.sqrt(3)*0.866*0.5],
        [-d*0.866, -d*np.sqrt(3)*0.866*0.5],
    ])
    return positions

def _ring_plus_positions():
    """Center + 6 ring + 5 more spread out."""
    d = np.sqrt(3)
    positions = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [-d, 0.0],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        # 5 more at larger radius
        [d*1.5, 0.0],
        [-d*1.5, 0.0],
        [0.0, d*np.sqrt(3)],
        [0.0, -d*np.sqrt(3)],
        [d*1.5, d*np.sqrt(3)/2],
    ])
    return positions

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s) where inner is (12,3) array, outer is (3,) array, s is side length.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    inits = []
    
    # Structured lattices
    inits.append(_hex_lattice_positions())
    inits.append(_compact_12_positions())
    inits.append(_ring_plus_positions())
    
    # Scaled versions
    for scale in [0.9, 0.95, 1.05, 1.1]:
        inits.append(_hex_lattice_positions() * scale)
    
    # Random perturbations of good structures
    for _ in range(8):
        base = _hex_lattice_positions()
        inits.append(base + np.random.randn(12, 2) * 0.3)
    
    for _ in range(5):
        base = _compact_12_positions()
        inits.append(base + np.random.randn(12, 2) * 0.2)
    
    # Random circular
    for r in [1.5, 2.0, 2.5, 3.0]:
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        pos = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        inits.append(pos)
    
    # Fully random
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 3.0)
    
    # Try multiple optimization strategies
    angles_to_try = [0, 30, 15, -15, 10, -10, 5, -5]
    
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Check if initial config is valid
        if not _check_separation(init_pos):
            # Try to fix by scaling down slightly
            scale = 0.95
            init_pos = init_pos * scale
            if not _check_separation(init_pos):
                continue
        
        for init_angle in angles_to_try[:4]:  # Limit angles for speed
            if time.time() > deadline - 20:
                break
            
            init_params = np.zeros(25)
            init_params[:24] = init_pos.flatten()
            init_params[24] = init_angle
            
            try:
                # Nelder-Mead for this non-smooth problem
                res = minimize(
                    _objective, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 3000, 
                        'xatol': 1e-9, 
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_s and res.fun < 100:
                    best_s = res.fun
                    best_params = res.x.copy()
                    
            except Exception:
                continue
    
    # Final polish: use the best result and do a high-precision refinement
    if best_params is not None and best_s < 100:
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    _objective, 
                    best_params, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 5000, 
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_s:
                    best_s = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Fallback if optimization failed
    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions()
        best_angle = 0
        best_s = _min_containing_hex_side(best_pos, best_angle, np.array([0.0, 0.0]))
        best_params = np.zeros(25)
        best_params[:24] = best_pos.flatten()
        best_params[24] = best_angle
    
    # Extract results
    final_positions = best_params[:24].reshape(12, 2)
    final_angle = best_params[24]
    final_center = np.array([0.0, 0.0])
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([final_center[0], final_center[1], final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
