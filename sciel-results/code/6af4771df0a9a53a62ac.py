# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_edge_normals(a_deg):
    """Compute 12 edge normals (6 for each hexagon, but we need separation check).
    For a unit hexagon with angle a, the 6 edge outward normals are at angles a+30+60k."""
    a = np.radians(a_deg)
    angles = a + np.radians(30.0 + 60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _check_separation(pos1, pos2, angle1, angle2):
    """Check if two unit hexagons are separated (not overlapping).
    Returns True if separated, False if overlapping."""
    # For two hexagons to be separated, on at least one of their 12 edge normals,
    # the projected intervals must be apart by more than 1e-6.
    # Edge normals for hex1 (6 normals) and hex2 (6 normals)
    n1 = _hex_edge_normals(angle1)  # (6, 2)
    n2 = _hex_edge_normals(angle2)  # (6, 2)
    
    # Project centers onto each normal
    d = pos2 - pos1  # vector from hex1 to hex2
    
    # For hex1's normals: the extent of hex1 along normal n is [-apothem, apothem] = [-sqrt(3)/2, sqrt(3)/2]
    # The extent of hex2 along normal n is [dot(pos2, n) - apothem, dot(pos2, n) + apothem]
    # They are separated along n if dot(d, n) > 2*apothem or dot(d, n) < -2*apothem
    apothem = np.sqrt(3) / 2
    
    for n in n1:
        proj = np.dot(d, n)
        if abs(proj) > 2 * apothem + 1e-6:
            return True  # Separated along this normal
    
    for n in n2:
        proj = np.dot(d, n)
        if abs(proj) > 2 * apothem + 1e-6:
            return True  # Separated along this normal
    
    return False  # Not separated (overlapping)

def _compute_enclosing_hex_side(positions, angles):
    """Compute the minimum side length of a regular hexagon that contains all vertices
    of the unit hexagons at given positions and angles. Returns (side, theta) where theta
    is the optimal rotation of the enclosing hexagon."""
    # Get all vertices of all hexagons
    all_vertices = []
    for i in range(len(positions)):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # (66, 2)
    
    # Find minimum enclosing hexagon
    # For a regular hexagon with center at origin, circumradius R, rotation theta:
    # The 6 edge normals are at angles theta+30+60k
    # The inradius (distance from center to edge) = R * cos(30°) = R * sqrt(3)/2
    # A point p is inside iff dot(p, n_k) <= R * sqrt(3)/2 for all k
    
    best_R = np.inf
    best_theta = 0.0
    
    # The function is periodic with period 60 degrees
    thetas = np.linspace(0, np.pi/3, 721)  # 0 to 60 degrees, 721 points for finer resolution
    for theta in thetas:
        normals = np.column_stack([
            np.cos(theta + np.radians(30 + 60*k)),
            np.sin(theta + np.radians(30 + 60*k))
        ] for k in range(6))
        # normals shape: (6, 2)
        dots = all_vertices @ normals.T  # (66, 6)
        max_dot = np.max(dots, axis=1)  # (66,)
        R_candidate = np.max(max_dot) / (np.sqrt(3)/2)
        if R_candidate < best_R:
            best_R = R_candidate
            best_theta = theta
    
    return best_R, best_theta

def _separation_penalty(positions, angles):
    """Compute a penalty for overlapping hexagons. Returns 0 if all separated."""
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(positions[i], positions[j], angles[i], angles[j]):
                # Compute how much they overlap
                d = positions[j] - positions[i]
                min_dist = 2 * np.sqrt(3) / 2  # minimum center distance for touching
                actual_dist = np.linalg.norm(d)
                if actual_dist < min_dist:
                    penalty += (min_dist - actual_dist) ** 2
                else:
                    penalty += 1.0  # They're close but not separated by normals
    return penalty

def _valid_configuration(positions, angles, outer_center, outer_angle, side):
    """Check if the configuration is valid according to the task requirements."""
    # Check all values finite
    if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(angles)) and 
            np.all(np.isfinite(outer_center)) and np.isfinite(outer_angle) and np.isfinite(side)):
        return False
    
    # Check side >= sqrt(11)
    if side < np.sqrt(11) - 1e-6:
        return False
    
    # Check separation between all pairs of inner hexagons
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(positions[i], positions[j], angles[i], angles[j]):
                return False
    
    # Check that all inner vertices are inside the outer hexagon
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    # For each outer edge, check that all inner vertices are on the correct side
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        edge_vec = p2 - p1
        # Normal pointing inward (toward center)
        normal = np.array([-edge_vec[1], edge_vec[0]])
        # Make sure normal points toward center
        center_to_p1 = outer_center - p1
        if np.dot(normal, center_to_p1) < 0:
            normal = -normal
        
        # Check all inner vertices
        for h in range(n):
            inner_verts = _hex_vertices(positions[h, 0], positions[h, 1], 1.0, angles[h])
            for v in inner_verts:
                # Cross product: cross(p2-p1, p-p1) should be >= -1e-6
                # This checks that p is to the left of the edge (inside)
                cross_val = np.cross(edge_vec, v - p1)
                if cross_val < -1e-6:
                    return False
    
    # Check edge lengths of inner hexagons are within 1e-8 of 1
    for h in range(n):
        verts = _hex_vertices(positions[h, 0], positions[h, 1], 1.0, angles[h])
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i+1)%6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # 580s deadline with margin
    
    # Initial configuration: honeycomb-like arrangement
    s3 = np.sqrt(3)
    
    # 7-hexagon cluster (1 + 6 neighbors)
    pos7 = np.array([
        [0, 0],
        [s3, 0],
        [s3/2, 1.5],
        [-s3/2, 1.5],
        [-s3, 0],
        [-s3/2, -1.5],
        [s3/2, -1.5],
    ])
    
    # Add 4 more to make 11
    # Place them in a way that keeps the cluster compact
    extra_pos = np.array([
        [s3*1.5, 0],
        [-s3*1.5, 0],
        [0, 3],
        [0, -3],
    ])
    
    positions = np.vstack([pos7, extra_pos])  # 11 positions
    angles = np.zeros(11)  # All hexagons have angle 0 initially
    
    # Objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pos = params[:22].reshape(11, 2)
        ang = params[22:33]
        
        # Compute separation penalty
        sep_pen = _separation_penalty(pos, ang)
        if sep_pen > 1e-4:
            return sep_pen * 1000  # Heavy penalty for overlap
        
        # Compute enclosing hexagon side
        side, theta = _compute_enclosing_hex_side(pos, ang)
        return side
    
    x0 = np.concatenate([positions.flatten(), angles])
    
    best_result = None
    best_val = np.inf
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    
    # Try a few random restarts
    np.random.seed(42)
    
    for restart in range(15):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(33) * 0.3
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if result.fun < best_val:
                best_val = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    # Extract best positions
    if best_result is not None:
        best_positions = best_result[:22].reshape(11, 2)
        best_angles = best_result[22:33]
    
    # Compute the optimal enclosing hexagon for the best configuration
    side, outer_theta = _compute_enclosing_hex_side(best_positions, best_angles)
    
    # Center the configuration at the origin for the outer hexagon
    # The outer hexagon should be centered at the centroid of the configuration
    centroid = np.mean(best_positions, axis=0)
    
    # Shift positions so centroid is at origin
    shifted_positions = best_positions - centroid
    
    # Recompute enclosing hexagon for shifted positions
    side_shifted, outer_theta_shifted = _compute_enclosing_hex_side(shifted_positions, best_angles)
    
    # Check if this configuration is valid
    outer_center = np.array([0.0, 0.0])
    outer_angle = outer_theta_shifted * 180 / np.pi  # Convert to degrees
    
    is_valid = _valid_configuration(shifted_positions, best_angles, outer_center, outer_angle, side_shifted)
    
    if not is_valid:
        # Fallback to a known valid configuration
        # Use the initial honeycomb arrangement with a larger enclosing hexagon
        shifted_positions = positions - np.mean(positions, axis=0)
        side_shifted, outer_theta_shifted = _compute_enclosing_hex_side(shifted_positions, angles)
        outer_center = np.array([0.0, 0.0])
        outer_angle = outer_theta_shifted * 180 / np.pi
        
        # If still invalid, use a very large hexagon
        if not _valid_configuration(shifted_positions, angles, outer_center, outer_angle, side_shifted):
            # Use a hexagon large enough to definitely contain everything
            max_dist = np.max(np.linalg.norm(shifted_positions, axis=1)) + 2
            side_shifted = max_dist * 2
            outer_angle = 0.0
            outer_center = np.array([0.0, 0.0])
    
    # Construct the output
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = shifted_positions[:, 0]
    inner_hex_data[:, 1] = shifted_positions[:, 1]
    inner_hex_data[:, 2] = best_angles  # angles in degrees
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, float(side_shifted)

# EVOLVE-BLOCK-END
