# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _minkowski_support(theta1, theta2):
    """
    Compute the support function of the Minkowski sum of two unit hexagons with angles theta1, theta2.
    The Minkowski sum of two unit hexagons is a dodecagon.
    Returns the 12 edge normals and the support values for quick overlap checking.
    """
    # For two unit hexagons with angles a1, a2, the Minkowski sum has 12 edges.
    # The edge normals are the union of edge normals from both hexagons (merged if parallel).
    # A unit hexagon with angle a has edge normals at angles a + 30 + 60*k for k=0..5
    # (since edges are perpendicular to the normals from center to edge midpoints)
    
    # Actually, let me think about this differently.
    # Two convex polygons P and Q are disjoint iff there exists a separating axis.
    # The candidate axes are the edge normals of P and Q.
    # For two unit hexagons, we have 12 candidate axes (6 from each).
    
    # But we can precompute: the Minkowski sum P+Q is a convex polygon with up to 12 edges.
    # P and Q are disjoint iff the origin is not in (P - Q) = P + (-Q).
    # So we need to check if origin is inside the Minkowski difference.
    
    # For quick checking, let's just use the 12 edge normals.
    a1 = np.radians(theta1)
    a2 = np.radians(theta2)
    
    # Edge normals of hexagon 1: directions perpendicular to edges
    # A hexagon with angle a has vertices at a + 60*k
    # Edges connect vertex k to vertex k+1
    # The outward normal of edge k is at angle a + 60*k + 30 (midpoint direction)
    normals1 = a1 + (np.arange(6) * np.pi / 3.0 + np.pi / 6.0)
    normals2 = a2 + (np.arange(6) * np.pi / 3.0 + np.pi / 6.0)
    
    all_normals = np.concatenate([normals1, normals2])
    all_normals = np.mod(all_normals, np.pi)  # Normals are defined up to pi
    
    # Remove near-duplicate normals
    unique_normals = []
    sorted_normals = np.sort(all_normals)
    for i in range(len(sorted_normals)):
        if i == 0 or np.abs(sorted_normals[i] - sorted_normals[i-1]) > 1e-6:
            unique_normals.append(sorted_normals[i])
    
    return np.array(unique_normals)

def _check_nonoverlap_fast(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Fast non-overlap check using precomputed normals.
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            # Get normals for both hexagons
            normals = _minkowski_support(angles_deg[i], angles_deg[j])
            
            overlap = True
            for normal_angle in normals:
                normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 + tol < min2 or max2 + tol < min1:
                    overlap = False
                    break
            if overlap:
                return False
    return True

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    normals_list = []
    for verts in vertices_list:
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=50.0):
    """
    Soft version of min outer side using log-sum-exp for smoothness.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # (72, 6)
    
    # For each direction, we need the max dot product
    # Use log-sum-exp for smooth max
    max_dot = np.zeros(6)
    for j in range(6):
        vals = dots[:, j]
        max_val = np.max(vals)
        logsumexp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        max_dot[j] = logsumexp / beta * beta  # This is just the logsumexp
        # Actually, let's use a proper soft-max
        max_dot[j] = max_val + np.log(np.sum(np.exp(vals - max_val))) / beta
    
    # Hmm, this isn't quite right. Let me use a simpler approach.
    # The actual max is what we want, but it's non-smooth.
    # Let's just use the hard max but with a smooth penalty for containment violations.
    
    global_max = np.max(dots)
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _objective_smooth(params, side=1.0, outer_angle_deg=0.0, beta=20.0):
    """
    Smooth objective: minimize outer hex side s with soft non-overlap penalty.
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    # Add smooth overlap penalty
    overlap_penalty = 0.0
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            # Compute minimum separation using support functions
            # For each of the 12 candidate normals, compute the gap
            normals = _minkowski_support(angles_deg[i], angles_deg[j])
            
            min_gap = np.inf
            for normal_angle in normals:
                normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                gap = min(np.max(proj2) - np.min(proj1), np.max(proj1) - np.min(proj2))
                min_gap = min(min_gap, gap)
            
            # If min_gap < 0, there's overlap
            if min_gap < 0:
                # Smooth penalty: use softplus or exponential
                overlap_penalty += np.exp(-min_gap * beta)
    
    return s + 10.0 * overlap_penalty

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    """
    Perform local refinement using scipy optimization with smooth objective.
    """
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    # Use scipy minimize with Nelder-Mead for robustness
    def obj(x):
        return _objective_smooth(x, side=1.0, outer_angle_deg=outer_angle_deg, beta=30.0)
    
    best_s = s
    best_params = params.copy()
    
    # Multiple restarts with different initial perturbations
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 30:
            break
        
        x0 = params.copy()
        if restart > 0:
            x0 += np.random.randn(36) * 0.1
        
        result = minimize(obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if result.success or result.fun < best_s:
            c_new = result.x.reshape(12, 3)[:, :2]
            a_new = result.x.reshape(12, 3)[:, 2]
            
            if _check_nonoverlap(c_new, a_new):
                verts_new = _compute_all_vertices(c_new, a_new)
                s_new = _min_outer_side(verts_new, outer_angle_deg)
                if s_new < best_s:
                    best_s = s_new
                    best_params = result.x.copy()
    
    return best_params, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill
    
    # Start with honeycomb arrangement
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)
        best_params = params.copy()
    
    # Global search: Simulated annealing with larger steps
    np.random.seed(42)
    T = 0.3
    cooling = 0.9995
    step_size = 0.2
    
    current_params = best_params.copy()
    current_s = best_s
    
    # Run until deadline or convergence
    max_iters = 300000
    for it in range(max_iters):
        if time.time() > deadline - 150:  # Reserve 150s for local refinement
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        
        if it % 1000 == 0 and it > 0:
            step_size *= 0.95
            step_size = np.clip(step_size, 0.05, 0.5)
    
    # Local refinement using scipy
    best_params, best_s = _local_refinement(best_params, deadline)
    
    # Final verification and cleanup
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    
    # Search for best outer hexagon angle
    best_outer_angle = 0.0
    s_final = _min_outer_side(all_verts_final, 0.0)
    
    # Coarse search over outer angles
    for angle_deg in np.arange(0, 60, 0.5):
        s_candidate = _min_outer_side(all_verts_final, angle_deg)
        if s_candidate < s_final:
            s_final = s_candidate
            best_outer_angle = angle_deg
    
    # Fine search around best angle
    if time.time() < deadline - 10:
        for angle_deg in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.01):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
        
        # Even finer search
        for angle_deg in np.arange(best_outer_angle - 0.01, best_outer_angle + 0.01, 0.001):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
    
    if not _check_nonoverlap(centers_final, angles_final):
        # Fallback to initial if somehow invalid
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
        best_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
