# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s):
    """Check validity of the packing configuration."""
    inner = np.array(inner, dtype=float)
    outer = np.array(outer, dtype=float)
    s = float(s)
    
    # Check finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Check outer vertices are distinct
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
    
    # Check unit edges of inner hexagons have length ~1
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(6):
            edge_len = np.linalg.norm(verts[j] - verts[(j+1)%6])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check no two inner hexagons overlap (separating axis test)
    for i in range(12):
        for j in range(i+1, 12):
            verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            verts_j = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            # Collect all edge normals from both hexagons
            normals = []
            for v in [verts_i, verts_j]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    # Normal to edge
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    normals.append(normal)
            
            # Check separating axis
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts_i, normal)
                proj_j = np.dot(verts_j, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # If intervals don't overlap, we have separation
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False
    
    # Check all inner vertices are inside or on the outer hexagon
    for i in range(12):
        verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for v in verts_i:
            # Check against each outer edge
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    inner = np.array(inner, dtype=float)
    outer_center = np.array(outer_center, dtype=float)
    
    # Shift inner centers by minus outer center
    shifted_centers = inner[:, :2] - outer_center
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Ensure normal points outward
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
            
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _generate_base_config():
    """
    Generate a valid base configuration of 12 hexagons centered at origin.
    This is a compact cluster: 1 center, 6 around it, 5 more?
    Actually, the previous program used a specific set. Let's use a standard hexagonal packing of 12.
    A hexagonal number for 12 is not an integer (1, 7, 19...).
    So we pack them in a 3x4 or similar grid.
    The seed from the prompt was:
    [0, 0, 0],
    [-2.5, 0, 0], [2.5, 0, 0],
    [-1.25, 2.165, 0], [1.25, 2.165, 0],
    [-1.25, -2.165, 0], [1.25, -2.165, 0],
    [-3.75, 2.165, 0], [3.75, 2.165, 0],
    [-3.75, -2.165, 0], [3.75, -2.165, 0],
    [0, -4.330, 0]
    This looks like a valid non-overlapping cluster.
    """
    initial_inner = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.1650635094610965, 0],
            [1.25, 2.1650635094610965, 0],
            [-1.25, -2.1650635094610965, 0],
            [1.25, -2.1650635094610965, 0],
            [-3.75, 2.1650635094610965, 0],
            [3.75, 2.1650635094610965, 0],
            [-3.75, -2.1650635094610965, 0],
            [3.75, -2.1650635094610965, 0],
            [0, -4.330127018922193, 0],
        ]
    )
    # Center of mass
    com = np.mean(initial_inner[:, :2], axis=0)
    initial_inner[:, :2] -= com
    return initial_inner

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses Differential Evolution on 4 parameters: (x, y, cluster_rotation, outer_rotation).
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    base_inner = _generate_base_config()
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e10
        
        x, y, phi, psi = params
        
        # Rotate and translate the base configuration
        phi_rad = np.radians(phi)
        cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
        
        new_inner = np.zeros_like(base_inner)
        new_inner[:, 0] = base_inner[:, 0] * cos_phi - base_inner[:, 1] * sin_phi + x
        new_inner[:, 1] = base_inner[:, 0] * sin_phi + base_inner[:, 1] * cos_phi + y
        new_inner[:, 2] = base_inner[:, 2] + phi
        
        outer_center = np.array([0.0, 0.0])
        s = _compute_min_outer_side(new_inner, outer_center, psi)
        
        return s

    # Bounds for the parameters
    # x, y: translation. The cluster extent is roughly +/- 5. So +/- 5 is safe.
    # phi: cluster rotation 0-360 (or -180 to 180)
    # psi: outer rotation 0-60 (due to symmetry)
    bounds = [(-5, 5), (-5, 5), (-180, 180), (0, 60)]
    
    best_result = None
    best_s = float('inf')
    
    try:
        # Differential Evolution is a global optimizer
        # maxiter=1000, popsize=15 (default)
        # We can increase popsize for better exploration if time permits, but 1000 iterations * 15 * 4 params is fast.
        result = differential_evolution(
            objective, bounds, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=True # Local refinement at the end
        )
        
        if result.fun < best_s:
            best_s = result.fun
            best_result = result.x
            
    except Exception:
        # Fallback if DE fails
        pass
    
    if best_result is None:
        # Fallback to base config with best outer angle
        best_psi = 0.0
        best_s = float('inf')
        for psi in np.linspace(0, 60, 12):
            s = _compute_min_outer_side(base_inner, np.array([0.0, 0.0]), psi)
            if s < best_s:
                best_s = s
                best_psi = psi
        best_result = np.array([0.0, 0.0, 0.0, best_psi])
    
    x, y, phi, psi = best_result
    
    # Construct final inner hexagons
    phi_rad = np.radians(phi)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    
    final_inner = np.zeros_like(base_inner)
    final_inner[:, 0] = base_inner[:, 0] * cos_phi - base_inner[:, 1] * sin_phi + x
    final_inner[:, 1] = base_inner[:, 0] * sin_phi + base_inner[:, 1] * cos_phi + y
    final_inner[:, 2] = base_inner[:, 2] + phi
    
    outer_center = np.array([0.0, 0.0])
    final_s = _compute_min_outer_side(final_inner, outer_center, psi)
    
    outer_hex_data = np.array([0.0, 0.0, psi])
    
    # Final validity check
    if not _is_valid(final_inner, outer_hex_data, final_s):
        # Fallback to base config with best psi
        final_inner = base_inner.copy()
        final_s = _compute_min_outer_side(final_inner, np.array([0.0, 0.0]), psi)
        if not _is_valid(final_inner, outer_hex_data, final_s):
            # Absolute fallback
            final_inner = _generate_base_config()
            final_s = _compute_min_outer_side(final_inner, np.array([0.0, 0.0]), 0.0)
    
    return final_inner, outer_hex_data, float(final_s)

# EVOLVE-BLOCK-END
