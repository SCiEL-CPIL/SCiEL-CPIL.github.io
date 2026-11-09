# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    """Compute outward unit normals for edges of a convex polygon given in CCW order."""
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _separation_check(verts1, normals1, verts2, normals2):
    """
    Check if two convex polygons are separated.
    Returns True if separated (valid), False if overlapping.
    Uses SAT: if any axis gives positive separation, they're separated.
    """
    all_normals = np.vstack([normals1, normals2])
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-6:
            return True  # Separated on this axis
        elif max2 < min1 - 1e-6:
            return True  # Separated on this axis
    
    return False  # No separating axis found with gap > 1e-6

def _compute_s_and_overlap_fast(inner, outer_angle):
    """
    Fast computation of required outer hexagon side length and overlap check.
    Returns (s, is_valid) where is_valid is True if no overlaps.
    """
    centers = inner[:, :2]
    angles = inner[:, 2]
    
    # Compute all inner vertices and normals
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    
    # Compute s: maximum projection onto outer hexagon outward normals
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    
    all_inner_verts_flat = all_verts.reshape(72, 2)
    
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    
    # Check overlaps using SAT
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            # Quick reject: if centers are far apart, definitely no overlap
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            
            if not _separation_check(all_verts[i], all_normals[i], all_verts[j], all_normals[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _generate_rigid_cluster():
    """
    Generate a fixed, rigid honeycomb cluster of 12 hexagons.
    Uses axial coordinates for a 'flower' (7) + 5 more.
    """
    # Axial coordinates (q, r)
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0)
    ]
    
    # Convert to Cartesian (flat-top hexagons, side 1)
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    pts = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append([x, y, 0.0])
        
    pts = np.array(pts)
    # Center the cluster
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    
    return pts

def _optimize_rigid_cluster():
    """
    Optimize the global rotation of the rigid cluster to minimize the enclosing hexagon size.
    """
    cluster = _generate_rigid_cluster()
    
    def objective(alpha):
        # Rotate cluster by alpha
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_pts = cluster[:, :2] @ rot_matrix.T
        
        # Compute max projection onto outer hexagon normals (angle 0)
        # Outer hexagon normals are at angles 0, 60, 120, 180, 240, 300
        # Wait, for flat-top hexagon, normals are at 0, 60, ...?
        # Vertices of unit flat-top hexagon: (1,0), (0.5, sqrt(3)/2), ...
        # Edge 0: (1,0) to (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4). Normal is (sqrt(3)/2, 0.5) -> angle 30 deg?
        # Let's just use the _edge_normals function on a unit hexagon.
        
        outer_verts = _hex_vertices(0, 0, 1.0, 0.0)
        outer_normals = _edge_normals(outer_verts)
        
        max_proj = 0.0
        for normal in outer_normals:
            projs = np.dot(rotated_pts, normal)
            max_k = np.max(projs)
            if max_k > max_proj:
                max_proj = max_k
        
        s = max_proj * 2.0 / np.sqrt(3)
        return s
    
    # Minimize in [0, pi/3]
    result = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-10})
    
    best_alpha = result.x
    best_s = result.fun
    
    # Rotate cluster by best_alpha
    cos_a = np.cos(best_alpha)
    sin_a = np.sin(best_alpha)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_pts = cluster[:, :2] @ rot_matrix.T
    
    # Construct inner array
    # Angles are also rotated
    inner_angles = np.full(12, np.degrees(best_alpha))
    inner = np.column_stack([rotated_pts, inner_angles])
    
    return inner, 0.0, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Phase 1: Rigid Cluster Optimization
    best_inner, best_outer_angle, best_s = _optimize_rigid_cluster()
    
    # Phase 2: Local Optimization (SA) starting from rigid cluster
    # This allows small deformations to potentially improve s further
    np.random.seed(42)
    
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.01
    T_min = 0.0001
    n_sa_iterations = 5000
    
    it = 0
    while it < n_sa_iterations:
        if time.time() > deadline - 60.0:
            break
        
        it += 1
        T = T_initial * (T_min / T_initial) ** (it / n_sa_iterations)
        
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb 1-2 hexagons
        n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * 0.02
            new_inner[idx, 1] += np.random.randn() * 0.02
            new_inner[idx, 2] += np.random.randn() * 0.01
        
        if np.random.rand() < 0.1:
            new_outer_angle += np.random.randn() * 0.01
        
        new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
        
        if new_valid:
            if new_s < current_s:
                current_inner = new_inner
                current_s = new_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        # Revert to rigid cluster if SA broke it
        best_inner, best_outer_angle, best_s = _optimize_rigid_cluster()
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
