# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(verts):
    """Return unit edge normals (6) for a hexagon's vertices in CCW order."""
    edges = np.roll(verts, -1, axis=0) - verts
    nx = edges[:, 1]
    ny = -edges[:, 0]
    norms = np.sqrt(nx ** 2 + ny ** 2)
    norms[norms < 1e-12] = 1.0
    return np.column_stack([nx / norms, ny / norms])


def _project_interval(verts, normal):
    """Project hexagon vertices onto a normal direction, return (min, max)."""
    dots = verts @ normal
    return dots.min(), dots.max()


def _hexagons_separated(v1, v2):
    """Check if two hexagons (each with 6 vertices) are separated by > 1e-6 gap."""
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for normal in all_normals:
        min1, max1 = _project_interval(v1, normal)
        min2, max2 = _project_interval(v2, normal)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False


def _all_inner_separated(inner_hexes):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_separated(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hexes, outer_verts):
    """Check all inner hex vertices are inside or on the outer hexagon."""
    n_outer = len(outer_verts)
    for hex_verts in inner_hexes:
        for p in hex_verts:
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def _valid_configuration(inner_data, outer_data, side):
    """Full validity check."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    ox, oy, oa = outer_data
    outer_verts = _hex_vertices(ox, oy, side, oa)

    inner_hexes = []
    for row in inner_data:
        cx, cy, ca = row
        verts = _hex_vertices(cx, cy, 1.0, ca)
        inner_hexes.append(verts)

    if not _all_inner_separated(inner_hexes):
        return False

    if not _inner_in_outer(inner_hexes, outer_verts):
        return False

    for hex_verts in inner_hexes:
        for i in range(6):
            d = np.linalg.norm(hex_verts[(i + 1) % 6] - hex_verts[i])
            if abs(d - 1.0) > 1e-8:
                return False

    return True


def _compute_required_side(inner_data, outer_center=None):
    """Compute the minimum side length of a regular hexagon (centered at outer_center or origin)
    that contains all inner hexagons. We optimize over the outer hexagon's angle."""
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    
    # Shift inner data so outer center is at origin
    shifted = inner_data.copy()
    shifted[:, 0] -= outer_center[0]
    shifted[:, 1] -= outer_center[1]
    
    # For a given outer hexagon angle, compute required side
    def required_side_for_angle(a_deg):
        a = np.radians(a_deg)
        # Outer hexagon edge normals (outward)
        normals = []
        for i in range(6):
            norm_angle = a + np.pi / 3 * (i + 1) - np.pi / 6
            normals.append(np.array([np.cos(norm_angle), np.sin(norm_angle)]))
        normals = np.array(normals)
        
        max_proj = 0.0
        for row in shifted:
            cx, cy, ca = row
            verts = _hex_vertices(cx, cy, 1.0, ca)
            for normal in normals:
                proj = np.max(verts @ normal)
                if proj > max_proj:
                    max_proj = proj
        
        return max_proj / (np.sqrt(3) / 2)
    
    # Try several angles to find the minimum required side
    min_side = np.inf
    for trial_angle in np.linspace(0, np.pi / 3, 13):
        s = required_side_for_angle(np.degrees(trial_angle))
        if s < min_side:
            min_side = s
    
    return min_side


def _compute_min_side_with_optimization(inner_data):
    """Use optimization to find the best outer hexagon angle and center."""
    # Center the configuration first
    centroid = np.mean(inner_data[:, :2], axis=0)
    centered = inner_data.copy()
    centered[:, 0] -= centroid[0]
    centered[:, 1] -= centroid[1]
    
    # Now find best angle
    def objective(params):
        angle_deg = params[0]
        cx, cy = params[1], params[2]
        # Shift centered data by (cx, cy) to account for outer center offset
        shifted = centered.copy()
        shifted[:, 0] -= cx
        shifted[:, 1] -= cy
        
        a = np.radians(angle_deg)
        normals = []
        for i in range(6):
            norm_angle = a + np.pi / 3 * (i + 1) - np.pi / 6
            normals.append(np.array([np.cos(norm_angle), np.sin(norm_angle)]))
        normals = np.array(normals)
        
        max_proj = 0.0
        for row in shifted:
            cx_h, cy_h, ca_h = row
            verts = _hex_vertices(cx_h, cy_h, 1.0, ca_h)
            for normal in normals:
                proj = np.max(verts @ normal)
                if proj > max_proj:
                    max_proj = proj
        
        return max_proj / (np.sqrt(3) / 2)
    
    # Try multiple initial angles
    best_side = np.inf
    for init_angle in np.linspace(0, np.pi / 3, 7):
        x0 = [np.degrees(init_angle), 0.0, 0.0]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
        except:
            pass
    
    return max(best_side, np.sqrt(11))


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit

    s3 = np.sqrt(3)

    # Generate multiple candidate configurations
    candidates = []
    
    # Configuration 1: Hexagonal lattice (center + 6 ring 1 + 4 from ring 2)
    positions = np.array([[0.0, 0.0, 0.0]])
    for i in range(6):
        angle = i * np.pi / 3
        positions = np.vstack([positions, [s3 * np.cos(angle), s3 * np.sin(angle), 0.0]])
    for i in range(4):
        angle = i * np.pi / 3 + np.pi / 6
        positions = np.vstack([positions, [2 * s3 * np.cos(angle), 2 * s3 * np.sin(angle), 0.0]])
    candidates.append(positions.copy())
    
    # Configuration 2: Different selection from ring 2
    positions = np.array([[0.0, 0.0, 0.0]])
    for i in range(6):
        angle = i * np.pi / 3
        positions = np.vstack([positions, [s3 * np.cos(angle), s3 * np.sin(angle), 0.0]])
    for i in [0, 2, 4, 6]:
        angle = i * np.pi / 6 + np.pi / 6
        positions = np.vstack([positions, [2 * s3 * np.cos(angle), 2 * s3 * np.sin(angle), 0.0]])
    candidates.append(positions.copy())
    
    # Configuration 3: Center + 6 ring 1 + 4 from ring 2 (different pattern)
    positions = np.array([[0.0, 0.0, 0.0]])
    for i in range(6):
        angle = i * np.pi / 3
        positions = np.vstack([positions, [s3 * np.cos(angle), s3 * np.sin(angle), 0.0]])
    for i in [1, 3, 5, 7]:
        angle = i * np.pi / 6 + np.pi / 6
        positions = np.vstack([positions, [2 * s3 * np.cos(angle), 2 * s3 * np.sin(angle), 0.0]])
    candidates.append(positions.copy())
    
    # Configuration 4: More spread out - center + 6 ring 1 + 4 at larger radius
    positions = np.array([[0.0, 0.0, 0.0]])
    for i in range(6):
        angle = i * np.pi / 3
        positions = np.vstack([positions, [s3 * np.cos(angle), s3 * np.sin(angle), 0.0]])
    for i in range(4):
        angle = i * np.pi / 2
        positions = np.vstack([positions, [2.2 * s3 * np.cos(angle), 2.2 * s3 * np.sin(angle), 0.0]])
    candidates.append(positions.copy())
    
    # Configuration 5: Two rows of 3 + center row of 5
    positions = []
    for x in [-2*s3, -s3, 0, s3, 2*s3]:
        positions.append([x, 0.0, 0.0])
    for x in [-1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3]:
        positions.append([x, s3*0.866, 0.0])
    positions.append([0.0, -s3*0.866, 0.0])
    candidates.append(np.array(positions))
    
    # Configuration 6: Triangular arrangement
    positions = []
    for i in range(3):
        for j in range(3):
            x = (i - 1) * s3 * 1.5
            y = (j - 1) * s3 * 0.866 + (i % 2) * s3 * 0.433
            positions.append([x, y, 0.0])
    positions.append([0.0, 0.0, 0.0])
    positions.append([s3, 0.0, 0.0])
    candidates.append(np.array(positions))
    
    best_side = np.inf
    best_inner = None
    
    for idx, candidate in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        
        # Center the configuration
        centroid = np.mean(candidate[:, :2], axis=0)
        centered = candidate.copy()
        centered[:, 0] -= centroid[0]
        centered[:, 1] -= centroid[1]
        
        # Check if inner hexagons are separated
        inner_hexes = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in centered]
        if not _all_inner_separated(inner_hexes):
            continue
        
        # Compute required side
        side = _compute_min_side_with_optimization(centered)
        
        if side < best_side:
            best_side = side
            best_inner = centered.copy()
        
        # Local optimization: try to improve positions
        current_positions = centered.copy()
        current_side = side
        
        rng = np.random.RandomState(idx * 100 + 42)
        step_size = 0.05
        n_local_iters = 3000
        
        for it in range(n_local_iters):
            if time.time() > deadline - 30:
                break
            
            # Try random perturbation of one hexagon
            idx_h = rng.randint(0, 11)
            delta = rng.normal(0, step_size, size=2)
            
            new_positions = current_positions.copy()
            new_positions[idx_h, :2] += delta
            
            # Check separation
            new_inner_hexes = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_positions]
            if not _all_inner_separated(new_inner_hexes):
                step_size *= 0.97
                continue
            
            new_side = _compute_required_side(new_positions)
            
            if new_side < current_side - 1e-8:
                current_positions = new_positions
                current_side = new_side
                step_size = min(step_size * 1.03, 0.15)
                
                if current_side < best_side:
                    best_side = current_side
                    best_inner = current_positions.copy()
            else:
                step_size *= 0.99
    
    # If no valid configuration found, use fallback
    if best_inner is None:
        best_inner = np.array([
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
        ])
        best_side = 8.0
    
    final_inner = best_inner.copy()
    final_side = max(best_side, np.sqrt(11))
    
    # Center the final configuration
    centroid = np.mean(final_inner[:, :2], axis=0)
    final_inner[:, 0] -= centroid[0]
    final_inner[:, 1] -= centroid[1]
    final_side = _compute_required_side(final_inner)
    final_side = max(final_side, np.sqrt(11))
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Final validity check
    if not _valid_configuration(final_inner, outer_data, final_side):
        # Fallback
        final_inner = np.array([
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
        ])
        outer_data = np.array([0, 0, 0])
        final_side = 8.0
    
    return final_inner, outer_data, final_side


# EVOLVE-BLOCK-END
