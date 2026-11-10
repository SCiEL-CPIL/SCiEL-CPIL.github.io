# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hex_vertices(cx, cy, s, a_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_validity(inner, outer, side):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11):
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = hex_vertices(ocx, ocy, side, oang)
    
    # Shift inner hexagons so outer center is at origin
    shifted_inner = inner.copy()
    shifted_inner[:, 0] -= ocx
    shifted_inner[:, 1] -= ocy
    
    # Check edge lengths of shifted inner hexagons
    for i in range(11):
        cx, cy, ang = shifted_inner[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between all pairs of inner hexagons
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = hex_vertices(shifted_inner[i, 0], shifted_inner[i, 1], 1.0, shifted_inner[i, 2])
            v2 = hex_vertices(shifted_inner[j, 0], shifted_inner[j, 1], 1.0, shifted_inner[j, 2])
            
            # Get all 12 edge normals (6 from each hexagon)
            normals = []
            for k in range(6):
                e1 = v1[(k + 1) % 6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                n1 /= (np.linalg.norm(n1) + 1e-15)
                normals.append(n1)
            
            for k in range(6):
                e2 = v2[(k + 1) % 6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                n2 /= (np.linalg.norm(n2) + 1e-15)
                normals.append(n2)
            
            # Check if any normal separates the two hexagons
            separated = False
            for n in normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                max1 = np.max(proj1)
                min1 = np.min(proj1)
                max2 = np.max(proj2)
                min2 = np.min(proj2)
                
                # Intervals are apart if max1 < min2 or max2 < min1
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check containment: all inner vertices must be inside outer hexagon
    for i in range(11):
        verts = hex_vertices(shifted_inner[i, 0], shifted_inner[i, 1], 1.0, shifted_inner[i, 2])
        for p in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def _compute_objective(params):
    """Compute the objective function (minimize side length)."""
    # params: 33 values for 11 hexagons (x, y, angle) each
    inner = params.reshape(11, 3)
    
    # Compute the minimum side length needed to contain all inner hexagons
    # We'll use a simple approach: find the bounding hexagon
    all_verts = []
    for i in range(11):
        cx, cy, ang = inner[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # Find center and max distance
    center = np.mean(all_verts, axis=0)
    distances = np.linalg.norm(all_verts - center, axis=1)
    max_dist = np.max(distances)
    
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # So we need s >= max_dist / (sqrt(3)/2) = 2*max_dist/sqrt(3)
    side = 2 * max_dist / np.sqrt(3)
    
    # Add a small margin for safety
    side *= 1.001
    
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Initial honeycomb lattice arrangement
    # Center hexagon at origin, 6 around it, 4 more
    hex_spacings = np.sqrt(3)  # Distance between centers of touching hexagons
    
    # Honeycomb lattice positions (using axial coordinates)
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance hex_spacings
    # Ring 2: 4 more hexagons
    
    initial_positions = np.array([
        [0, 0, 0],  # center
        [hex_spacings, 0, 0],  # right
        [hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # top-right
        [-hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # top-left
        [-hex_spacings, 0, 0],  # left
        [-hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # bottom-left
        [hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # bottom-right
        [3*hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # far top-right
        [3*hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # far bottom-right
        [-3*hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # far top-left
        [-3*hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # far bottom-left
    ])
    
    # Flatten for optimization
    x0 = initial_positions.flatten()
    
    # Optimize to minimize the side length
    best_params = x0.copy()
    best_side = _compute_objective(best_params)
    
    # Try Nelder-Mead optimization
    if time.time() < deadline - 300:
        try:
            result = minimize(
                _compute_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except:
            pass
    
    # Polish with more iterations
    if time.time() < deadline - 100:
        try:
            result = minimize(
                _compute_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except:
            pass
    
    # Construct the final result
    inner = best_params.reshape(11, 3)
    
    # Compute the actual minimum side length needed
    all_verts = []
    for i in range(11):
        cx, cy, ang = inner[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    center = np.mean(all_verts, axis=0)
    distances = np.linalg.norm(all_verts - center, axis=1)
    max_dist = np.max(distances)
    
    # Side length for regular hexagon
    side = 2 * max_dist / np.sqrt(3)
    side *= 1.002  # Small margin
    
    # Center the outer hexagon
    outer = np.array([center[0], center[1], 0.0])
    
    # Shift inner hexagons so outer center is at origin
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= center[0]
    inner_shifted[:, 1] -= center[1]
    
    # Verify validity
    if not _check_validity(inner_shifted, np.array([0, 0, 0]), side):
        # Fallback to the initial arrangement
        inner_shifted = initial_positions.copy()
        all_verts = []
        for i in range(11):
            cx, cy, ang = inner_shifted[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        center = np.mean(all_verts, axis=0)
        distances = np.linalg.norm(all_verts - center, axis=1)
        max_dist = np.max(distances)
        side = 2 * max_dist / np.sqrt(3) * 1.002
        inner_shifted[:, 0] -= center[0]
        inner_shifted[:, 1] -= center[1]
    
    return inner_shifted, np.array([0, 0, 0]), side


# EVOLVE-BLOCK-END
