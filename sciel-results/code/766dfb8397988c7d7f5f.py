# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_pair_separation(hex1, hex2, tol=1e-6):
    """
    Check if two unit hexagons are separated (non-intersecting).
    Returns True if they are separated on at least one of the 12 edge normals.
    """
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get all 12 edge normals (6 from each hexagon)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
                normals.append(n)
    
    # Check each normal
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if intervals are apart by more than tol
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False


def _check_inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon.
    For each inner vertex, check cross product with each outer edge.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for hex_data in inner_hexes:
        inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
        
        for iv in inner_verts:
            # Shift by outer center
            iv_shifted = iv - outer_center
            
            for i in range(6):
                p1 = outer_verts[i] - outer_center
                p2 = outer_verts[(i + 1) % 6] - outer_center
                
                # Cross product: (p2 - p1) x (p - p1)
                edge = p2 - p1
                vec = iv_shifted - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]
                
                if cross < -tol:
                    return False
    
    return True


def _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle):
    """
    Find the minimum outer side length such that all inner hexagons fit inside.
    """
    # For each inner vertex, compute the required distance from center
    # along each of the 6 outer hexagon edge directions
    
    # The outer hexagon with side s has inradius = s * sqrt(3)/2
    # For a point to be inside, its projection onto each edge normal must be <= inradius
    
    min_side = 0.0
    
    # The 6 edge normals of a regular hexagon centered at origin with angle 0
    # Point outward from each edge
    for i in range(6):
        angle = np.radians(60 * i + 30)  # Normal to edge i
        n = np.array([np.cos(angle), np.sin(angle)])
        
        max_proj = 0.0
        for hex_data in inner_hexes:
            inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
            # Shift by outer center
            inner_verts_shifted = inner_verts - outer_center
            projs = inner_verts_shifted @ n
            max_proj = max(max_proj, np.max(projs))
        
        # The inradius must be at least max_proj
        # inradius = s * sqrt(3)/2, so s >= 2 * max_proj / sqrt(3)
        required_side = 2.0 * max_proj / np.sqrt(3.0)
        min_side = max(min_side, required_side)
    
    return min_side


def _objective(params, outer_center, outer_angle):
    """
    Objective function: minimize outer side length.
    params: flat array of 11*3 = 33 values (x, y, angle for each inner hex)
    """
    inner_hexes = params.reshape(11, 3)
    
    # Check pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                return 1e6  # Large penalty for overlap
    
    # Compute minimum outer side
    side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
    
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Outer hexagon centered at origin with angle 0
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Initial guess: hexagonal arrangement
    # Center hex at origin, 6 around it, 4 more
    sqrt3 = np.sqrt(3)
    initial_positions = [
        [0, 0, 0],           # center
        [3, 0, 0],           # right
        [-3, 0, 0],          # left
        [1.5, 2.598, 0],     # top-right (3/2, 3*sqrt(3)/2)
        [-1.5, 2.598, 0],    # top-left
        [1.5, -2.598, 0],    # bottom-right
        [-1.5, -2.598, 0],   # bottom-left
        [4.5, 2.598, 0],     # far top-right
        [-4.5, 2.598, 0],    # far top-left
        [4.5, -2.598, 0],    # far bottom-right
        [-4.5, -2.598, 0],   # far bottom-left
    ]
    
    initial_params = np.array(initial_positions).flatten()
    
    # Try multiple optimization runs with different initial guesses
    best_params = initial_params.copy()
    best_side = _min_outer_side_for_inner(initial_params.reshape(11, 3), outer_center, outer_angle)
    
    # Run optimization
    n_runs = 5
    for run in range(n_runs):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        # Add small random perturbation for different runs
        if run == 0:
            params0 = initial_params.copy()
        else:
            params0 = initial_params + np.random.randn(33) * 0.5
        
        try:
            result = minimize(
                _objective,
                params0,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if result.success or result.fun < best_side:
                # Verify the solution is valid
                inner_hexes = result.x.reshape(11, 3)
                
                # Check pairwise separation
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_params = result.x.copy()
                        
        except Exception:
            continue
    
    # Final result
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
