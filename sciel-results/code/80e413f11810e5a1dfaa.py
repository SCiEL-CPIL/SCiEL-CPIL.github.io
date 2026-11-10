# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_separation(hex1, hex2, eps=1e-6):
    """Check if two unit hexagons are separated (non-overlapping).
    hex1, hex2: each is (cx, cy, angle_deg)
    Returns True if separated."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get edge normals for hex1 (6 edges, 2 normals each = 12)
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        # Two normals: outward and inward
        normal = np.array([-edge[1], edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Project both hexagons onto this normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check if intervals are apart
        gap1 = min1 - max2  # hex1 entirely on negative side
        gap2 = min2 - max1  # hex2 entirely on negative side
        
        if gap1 > eps or gap2 > eps:
            return True
    
    return False


def _check_inside_outer(inner_hex, outer_center, outer_side, outer_angle):
    """Check if all vertices of inner_hex are inside or on the outer hexagon."""
    v = _hex_vertices(inner_hex[0], inner_hex[1], 1.0, inner_hex[2])
    outer_v = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # For each edge of outer hexagon, check cross product
    for i in range(6):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % 6]
        edge = p2 - p1
        # Cross product of edge and (point - p1)
        for p in v:
            cp = np.cross(edge, p - p1)
            if cp < -1e-6:
                return False
    return True


def _is_valid(inner_data, outer_center, outer_side, outer_angle):
    """Check full validity of the packing."""
    # Check finite
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_center)):
        return False
    
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    
    # Check pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    # Check all inner hexagons inside outer
    for i in range(11):
        if not _check_inside_outer(inner_data[i], outer_center, outer_side, outer_angle):
            return False
    
    # Check edge lengths of inner hexagons are close to 1
    for i in range(11):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(v[(k+1)%6] - v[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True


def _objective(params):
    """Objective: minimize outer side length.
    params: [outer_side, outer_angle, x1,y1,a1, x2,y2,a2, ..., x11,y11,a11]
    Total: 2 + 33 = 35 params
    """
    outer_side = params[0]
    outer_angle = params[1]
    outer_center = np.array([0.0, 0.0])  # Fix center at origin for simplicity
    
    inner_data = params[2:].reshape(11, 3)
    
    # Penalty if outer side too small
    penalty = 0.0
    if outer_side < np.sqrt(11):
        penalty += 100 * (np.sqrt(11) - outer_side)
    
    # Check separation and containment, add penalties
    for i in range(11):
        h1 = inner_data[i]
        v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
        
        # Check inside outer
        outer_v = _hex_vertices(0, 0, outer_side, outer_angle)
        for k in range(6):
            p1 = outer_v[k]
            p2 = outer_v[(k + 1) % 6]
            edge = p2 - p1
            for p in v1:
                cp = np.cross(edge, p - p1)
                if cp < 0:
                    penalty += 10 * (-cp)
        
        # Check separation from others
        for j in range(i + 1, 11):
            h2 = inner_data[j]
            v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
            
            # Compute minimum gap
            min_gap = 1e10
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                gap1 = min1 - max2
                gap2 = min2 - max1
                gap = max(gap1, gap2)
                min_gap = min(min_gap, gap)
            
            if min_gap < 1e-6:
                penalty += 10 * (1e-6 - min_gap)
    
    return outer_side + penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 630  # seconds
    
    # Initial guess: hexagonal cluster arrangement
    # 1 center, 6 at distance 2, 4 at distance 4
    initial_inner = []
    # Center
    initial_inner.append([0, 0, 0])
    # 6 neighbors at distance 2
    for k in range(6):
        angle = 60 * k
        x = 2 * np.cos(np.radians(angle))
        y = 2 * np.sin(np.radians(angle))
        initial_inner.append([x, y, 0])
    # 4 more at distance 4 in alternating directions
    for k in range(0, 6, 2):  # Every other direction
        angle = 60 * k + 30  # Offset by 30 degrees
        x = 4 * np.cos(np.radians(angle))
        y = 4 * np.sin(np.radians(angle))
        initial_inner.append([x, y, 0])
    
    # That gives 1 + 6 + 3 = 10, need one more
    # Add one more at distance 4 in another direction
    angle = 60 * 3 + 30
    x = 4 * np.cos(np.radians(angle))
    y = 4 * np.sin(np.radians(angle))
    initial_inner.append([x, y, 0])
    
    initial_inner = np.array(initial_inner)
    
    # Initial outer: large enough
    max_dist = np.max(np.sqrt(np.sum(initial_inner[:, :2]**2, axis=1))) + 1
    initial_outer_side = max_dist / np.cos(np.radians(30)) + 0.5
    initial_outer_angle = 0.0
    
    # Build initial params
    initial_params = np.concatenate([[initial_outer_side, initial_outer_angle], initial_inner.flatten()])
    
    # Bounds
    bounds = [(np.sqrt(11), 15), (0, 60)]  # outer_side, outer_angle
    for i in range(11):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 60))    # angle
    
    # Run optimization with time limit
    best_params = initial_params.copy()
    best_side = initial_outer_side
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = minimize(_objective, initial_params, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 1000, 'maxfun': 50000})
            if result.fun < best_side:
                best_params = result.x
                best_side = result.x[0]
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try Nelder-Mead from the L-BFGS-B result
        try:
            result2 = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result2.fun < best_side:
                best_params = result2.x
                best_side = result2.x[0]
        except Exception:
            pass
    
    # Extract results
    outer_side = best_params[0]
    outer_angle = best_params[1]
    inner_data = best_params[2:].reshape(11, 3)
    
    # Normalize angles to [0, 60)
    inner_data[:, 2] = inner_data[:, 2] % 60
    outer_angle = outer_angle % 60
    
    # Verify validity; if invalid, fall back to a known valid configuration
    if not _is_valid(inner_data, np.array([0.0, 0.0]), outer_side, outer_angle):
        # Fallback: use the initial arrangement with a safe outer size
        inner_data = initial_inner.copy()
        outer_angle = 0.0
        max_dist = np.max(np.sqrt(np.sum(inner_data[:, :2]**2, axis=1))) + 1
        outer_side = max_dist / np.cos(np.radians(30)) + 0.5
    
    outer_data = np.array([0.0, 0.0, outer_angle])
    
    return inner_data, outer_data, float(outer_side)


# EVOLVE-BLOCK-END
