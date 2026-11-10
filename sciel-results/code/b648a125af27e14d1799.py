# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_hexes, angle_deg=0):
    """
    Find minimum side length of a regular hexagon (at given angle, centered at origin)
    that contains all inner hexagons.
    """
    a = np.radians(angle_deg)
    # The 6 edge normals of the outer hexagon
    normals = np.column_stack([
        np.cos(a + np.radians(60) * np.arange(6)),
        np.sin(a + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0.0
    for hex_data in inner_hexes:
        cx, cy, angle = hex_data
        verts = _hex_vertices(cx, cy, 1.0, angle)
        for normal in normals:
            projections = verts @ normal
            max_proj = max(max_proj, np.max(projections))
    
    # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
    # The max projection onto a normal must be <= apothem
    # So s >= max_proj / (sqrt(3)/2)
    return max_proj / (np.sqrt(3) / 2)


def _check_separation(inner_hexes):
    """Check that no two inner hexagons overlap using separating axis test."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            hex_i = _hex_vertices(*inner_hexes[i])
            hex_j = _hex_vertices(*inner_hexes[j])
            
            # Get all edge normals from both hexagons
            all_normals = []
            for hex_verts in [hex_i, hex_j]:
                for k in range(6):
                    p1 = hex_verts[k]
                    p2 = hex_verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    all_normals.append(normal)
            
            # Check each normal
            for normal in all_normals:
                proj_i = hex_i @ normal
                proj_j = hex_j @ normal
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # If there's a gap, they're separated on this axis
                if max_i + 1e-6 < min_j or max_j + 1e-6 < min_i:
                    break
                else:
                    # Check if they overlap on this axis
                    if max_i >= min_j and max_j >= min_i:
                        # They overlap on this axis, continue checking other axes
                        continue
                    else:
                        break
            else:
                # No separating axis found - hexagons overlap
                return False
    return True


def _objective(inner_hexes, outer_angle_deg=0):
    """Compute 1/s where s is the minimum outer hexagon side length."""
    s = _min_containing_hex_side(inner_hexes, outer_angle_deg)
    return 1.0 / s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 630.0 - 30.0  # 30s safety margin
    
    # Initial arrangement: hexagonal grid pattern
    # 12 hexagons in a compact arrangement
    # Pattern: 3-4-3 or similar compact layout
    
    # Let's use a more optimized initial arrangement
    # Hexagonal grid with spacing 2 (touching)
    sqrt3 = np.sqrt(3)
    
    # Arrangement: center + 6 around + 5 more
    # Using axial coordinates for hex grid
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance 2
    # Ring 2: 6 hexagons at distance 4 (but we only need 5 more to total 12)
    
    # Better: Use a 3x4 rectangular-ish arrangement optimized for hexagonal container
    
    # Let me try a symmetric arrangement
    # 12 hexagons: can be arranged as 2-3-3-2-2 or similar
    
    # Optimal known: For 12 unit hexagons in a hexagon, the minimal container
    # has side length around 5-6
    
    # Let's start with a reasonable arrangement and optimize
    inner_hex_data = np.array([
        [0, 0, 0],           # 0: center
        [-2, 0, 0],          # 1: left
        [2, 0, 0],           # 2: right
        [-1, sqrt3, 0],      # 3: top-left
        [1, sqrt3, 0],       # 4: top-right
        [-1, -sqrt3, 0],     # 5: bottom-left
        [1, -sqrt3, 0],      # 6: bottom-right
        [-3, sqrt3, 0],      # 7: far top-left
        [3, sqrt3, 0],       # 8: far top-right
        [-3, -sqrt3, 0],     # 9: far bottom-left
        [3, -sqrt3, 0],      # 10: far bottom-right
        [0, 2*sqrt3, 0],     # 11: top
    ])
    
    # Verify initial arrangement is valid
    if not _check_separation(inner_hex_data):
        # Fallback to simpler arrangement
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2, 0, 0],
            [2, 0, 0],
            [-1, sqrt3, 0],
            [1, sqrt3, 0],
            [-1, -sqrt3, 0],
            [1, -sqrt3, 0],
            [-3, 0, 0],
            [3, 0, 0],
            [-2, 2*sqrt3, 0],
            [2, 2*sqrt3, 0],
            [0, -2*sqrt3, 0],
        ])
    
    best_inner = inner_hex_data.copy()
    best_score = _objective(best_inner, 0)
    
    # Optimize: try different outer hexagon angles
    for angle in [0, 30, 60, 90, 120, 150]:
        score = _objective(inner_hex_data, angle)
        if score > best_score:
            best_score = score
    
    # Simple local optimization: perturb positions
    np.random.seed(42)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        candidate = best_inner.copy()
        idx = np.random.randint(12)
        candidate[idx, 0] += np.random.uniform(-0.1, 0.1)
        candidate[idx, 1] += np.random.uniform(-0.1, 0.1)
        
        # Check separation
        if _check_separation(candidate):
            score = _objective(candidate, 0)
            if score > best_score:
                best_score = score
                best_inner = candidate
    
    # Find optimal outer angle for best arrangement
    best_outer_angle = 0
    for angle in np.linspace(0, 180, 37):
        score = _objective(best_inner, angle)
        if score > best_score:
            best_score = score
            best_outer_angle = angle
    
    outer_hex_data = np.array([0, 0, best_outer_angle])
    outer_hex_side_length = 1.0 / best_score
    
    return best_inner, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
