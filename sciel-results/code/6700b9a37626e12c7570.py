import numpy as np
import time
import math
from scipy.optimize import linprog, minimize

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, np.array([cx, cy])

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _hex_min_distance(c1, a1, c2, a2):
    """
    Compute the minimum distance between two hexagons (0 if overlapping).
    Uses SAT to find the minimum separation.
    """
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    min_sep = float('inf')
    
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep > min_sep:
            min_sep = sep
    
    return max(0.0, min_sep)

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _optimize_outer(centers, inner_angles, fine=False):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep
    n_coarse = 25
    for outer_angle in np.linspace(0, 60, n_coarse):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # Fine local refinement around best angle
    if fine:
        n_fine = 41
        for outer_angle in np.linspace(best_angle - 3.0, best_angle + 3.0, n_fine):
            if outer_angle < 0 or outer_angle > 60:
                continue
            side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _try_arrangement(centers, inner_angles=None, fine=False):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles, fine=fine)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    x = 0
    y = 0
    for i in range(N):
        centers.append((x, y))
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0
    
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _smooth_overlap_penalty(centers, angles, weight=10.0):
    """
    Compute a smooth penalty for overlapping hexagons.
    Returns total penalty (0 if no overlaps).
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = _hex_min_distance(centers[i], angles[i], centers[j], angles[j])
            if dist < 0.01:
                # Quadratic penalty for near-overlap
                penalty += weight * (0.01 - dist) ** 2
    return penalty

def _contact_graph_optimize(centers, angles, outer_side, outer_center, outer_angle, max_iter=200):
    """
    Optimize using contact graph / force-directed approach with smooth penalties.
    """
    centers = [list(c) for c in centers]
    angles = list(angles)
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle)
    inradius = outer_side * sqrt3 / 2
    
    for it in range(max_iter):
        # Compute gradients
        grad_centers = np.zeros((N, 2))
        grad_angles = np.zeros(N)
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                dist = _hex_min_distance(centers[i], angles[i], centers[j], angles[j])
                if dist < 0.1:
                    # Push apart
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > 1e-12:
                        dx /= d
                        dy /= d
                        force = 1.0 / (0.1 + dist)
                        grad_centers[i][0] -= dx * force
                        grad_centers[i][1] -= dy * force
                        grad_centers[j][0] += dx * force
                        grad_centers[j][1] += dy * force
        
        # Containment: push hexagons inside if they extend outside
        for i in range(N):
            verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    d = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                    if d > inradius:
                        grad_centers[i][0] -= nx * (d - inradius) * 10.0
                        grad_centers[i][1] -= ny * (d - inradius) * 10.0
        
        # Apply gradient descent
        lr = 0.01
        for i in range(N):
            centers[i][0] -= lr * grad_centers[i][0]
            centers[i][1] -= lr * grad_centers[i][1]
    
    return centers, angles

def _polish(centers, angles, outer_side, outer_center, outer_angle, max_time=10.0):
    """
    High-precision polish: minimize outer side with smooth constraints.
    Uses L-BFGS-B with a smooth objective that penalizes violations.
    """
    start_time = time.time()
    centers = np.array(centers, dtype=float)
    angles = np.array(angles, dtype=float)
    sqrt3 = np.sqrt(3)
    
    def smooth_objective(x):
        if time.time() - start_time > max_time:
            return float('inf')
        
        c = np.array(x[:2*N]).reshape(N, 2)
        a = x[2*N:]
        
        # Compute outer hexagon for current positions
        side, center, angle = _optimize_outer(c, a, fine=False)
        if not np.isfinite(side):
            return 1e6
        
        outer_normals = _hex_edge_normals(angle)
        inradius = side * sqrt3 / 2
        
        # Containment penalty (smooth)
        penalty = 0.0
        for i in range(N):
            verts = _hex_vertices(c[i][0], c[i][1], 1.0, a[i])
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    d = (vx - center[0]) * nx + (vy - center[1]) * ny
                    if d > inradius:
                        penalty += (d - inradius) ** 2
        
        # Overlap penalty (smooth)
        for i in range(N):
            for j in range(i + 1, N):
                dist = _hex_min_distance(c[i], a[i], c[j], a[j])
                if dist < 0.05:
                    penalty += 10.0 * (0.05 - dist) ** 2
        
        return side + penalty
    
    x0 = np.concatenate([centers.flatten(), angles])
    
    try:
        res = minimize(smooth_objective, x0, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10, 'maxfun': 5000})
        
        if np.isfinite(res.fun) and res.fun < 1e5:
            c = np.array(res.x[:2*N]).reshape(N, 2)
            a = res.x[2*N:]
            centers_list = [list(c[i]) for i in range(N)]
            angles_list = list(a)
            return centers_list, angles_list
    except Exception:
        pass
    
    return [list(c) for c in centers], list(angles)

def _soft_max(values, temperature=0.01):
    """
    Differentiable approximation of max.
    """
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_vals)
    return temperature * (max_val + np.log(sum_exp))

def _smooth_outer_side(centers, angles, outer_angle_deg):
    """
    Compute smooth outer side length and center for fixed outer angle.
    Returns (side, center_x, center_y).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
    
    # We need to solve for center (cx, cy) and inradius r such that
    # n_i . (c - p) <= r for all points p and normals n_i.
    # Equivalent to: n_i . c <= r + n_i . p for all p.
    # Let A_i = max_p (n_i . p). Then n_i . c - r <= -A_i.
    # Minimize r subject to n_i . c - r <= -A_i.
    
    # To make this differentiable w.r.t centers, we use soft-max for A_i.
    # A_i = max_p (n_i . p) approx soft_max_p (n_i . p).
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) 
                            for i in range(len(centers))])
    
    # Projections shape (num_points, 6)
    projections = all_points @ normals.T
    
    # A_i = soft_max over points for each normal i
    # We want gradient of A_i w.r.t centers.
    # Instead of full autodiff, we can use the fact that for a fixed center,
    # the optimal r is max_i (A_i - n_i . c).
    # And optimal c minimizes this max.
    # This is a convex problem.
    
    # For a smooth approximation, we can define the objective as:
    # f(c) = soft_max_i (A_i - n_i . c)
    # The minimum of f(c) over c gives the inradius r.
    # The optimal c is where the gradient of f(c) is 0.
    # Since f is convex, we can just evaluate the minimum value.
    # However, we need the value of r (and c) to return.
    
    # Let's find the optimal c using a few steps of gradient descent on the smooth objective,
    # or simply use the LP solution but smooth the A_i inputs.
    # Actually, for the polish, we can just use the LP solver which is fast,
    # but the objective needs to be smooth w.r.t centers.
    # The LP solution r* is a convex function of A_i, which is a convex function of centers.
    # So r* is convex w.r.t centers.
    # We can just compute r* using LP and return it. The gradient will be handled by L-BFGS-B finite differences.
    
    # To speed up LP, we can reuse the structure.
    A = projections.max(axis=0)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), 0.0, 0.0
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, cx, cy

def _gradient_polish(centers, angles, outer_angle_deg, max_time=20.0):
    """
    Fast gradient polish using L-BFGS-B and smooth objective.
    """
    start_time = time.time()
    centers = np.array(centers, dtype=float)
    angles = np.array(angles, dtype=float)
    sqrt3 = np.sqrt(3)
    
    def objective(x):
        if time.time() - start_time > max_time:
            return float('inf')
        
        c = np.array(x[:2*N]).reshape(N, 2)
        a = x[2*N:]
        
        side, cx, cy = _smooth_outer_side(c, a, outer_angle_deg)
        if not np.isfinite(side):
            return 1e6
        
        # Penalty for overlaps (smooth approximation)
        # Using center distance as a proxy for hexagon distance is rough but fast.
        # Better: use min distance if it's not too slow.
        # For N=11, 55 pairs. _hex_min_distance is SAT-based, might be slow.
        # Let's use a smooth penalty based on center distance and rotation.
        # Approximation: if center distance < 2*sqrt(3) (touching distance for aligned hexes), penalize.
        # But rotations change this.
        # Let's stick to _hex_min_distance but only if we have time.
        # Actually, let's use a simpler penalty: sum of squared distances for pairs that are close.
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i][0] - c[j][0]
                dy = c[i][1] - c[j][1]
                dist_sq = dx*dx + dy*dy
                # Threshold: 2*sqrt(3) ~ 3.46 is min distance for aligned.
                # With rotation, min distance can be smaller.
                # Let's use a threshold of 3.0 to be safe.
                threshold = 3.0
                if dist_sq < threshold * threshold:
                    # Smooth penalty
                    dist = np.sqrt(dist_sq)
                    penalty += (threshold - dist) ** 2
        
        return side + 10.0 * penalty
    
    x0 = np.concatenate([centers.flatten(), angles])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 2000})
        
        if np.isfinite(res.fun) and res.fun < 1e5:
            c = np.array(res.x[:2*N]).reshape(N, 2)
            a = res.x[2*N:]
            centers_list = [list(c[i]) for i in range(N)]
            angles_list = list(a)
            return centers_list, angles_list
    except Exception:
        pass
    
    return [list(c) for c in centers], list(angles)

def _generate_initial_configurations():
    """Generate diverse initial configurations for multi-start."""
    configs = []
    
    # Config 1: Hexagonal lattice (center + 6 ring + 4 from second ring)
    d1 = 2 * np.sqrt(3)
    d2 = 4 * np.sqrt(3)
    first_ring = [(d1 * np.cos(np.pi/3*i), d1 * np.sin(np.pi/3*i)) for i in range(6)]
    second_ring = []
    for i in range(6):
        a = np.pi/3*i
        second_ring.append((d2 * np.cos(a), d2 * np.sin(a)))
        a2 = np.pi/3*i + np.pi/6
        second_ring.append((d2 * np.cos(a2), d2 * np.sin(a2)))
    
    # Try different subsets of second ring
    subsets = [
        [0, 2, 4, 6],
        [0, 3, 6, 9],
        [1, 3, 5, 7],
        [0, 1, 2, 3],
        [0, 2, 5, 7],
        [0, 4, 8, 10],
    ]
    for subset in subsets:
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        configs.append((centers, [0.0] * N))
    
    # Config 2: Linear arrangement with slight zigzag
    for spacing in [3.5, 3.8, 4.0]:
        centers = []
        for i in range(N):
            x = (i - N//2) * spacing
            y = 0.5 * spacing * (1 if i % 2 == 0 else -1)
            centers.append((x, y))
        configs.append((centers, [0.0] * N))
    
    # Config 3: Circular arrangement
    for radius in [2.5, 3.0, 3.5]:
        centers = [(0.0, 0.0)]
        for i in range(N - 1):
            angle = 2 * np.pi * i / (N - 1)
            centers.append((radius * np.cos(angle), radius * np.sin(angle)))
        configs.append((centers, [0.0] * N))
    
    # Config 4: Random with seed
    np.random.seed(123)
    for _ in range(3):
        centers = [(0.0, 0.0)]
        for i in range(N - 1):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(1.5, 3.5)
            centers.append((r * np.cos(angle), r * np.sin(angle)))
        configs.append((centers, [0.0] * N))
    
    return configs

def _simulate_annealing(initial_centers, initial_angles, max_time=60.0):
    """
    Simulated Annealing to optimize positions and angles.
    """
    start_time = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    
    # Initial cost
    current_side, current_center, current_angle = _optimize_outer(centers, angles)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None, None
        
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center
    best_angle = current_angle
    
    T = 0.3
    T_min = 0.005
    cooling_rate = 0.998
    
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Generate new state
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Random perturbation - smaller steps for better local search
        idx = np.random.randint(N)
        scale = 0.05 + 0.1 * (T / 0.3)
        delta_x = np.random.normal(0, scale)
        delta_y = np.random.normal(0, scale)
        new_centers[idx][0] += delta_x
        new_centers[idx][1] += delta_y
        
        if np.random.rand() < 0.5:
            rot_idx = np.random.randint(N)
            delta_ang = np.random.normal(0, 3.0 * (T / 0.3))
            new_angles[rot_idx] += delta_ang
            
        if np.random.rand() < 0.05:
            dx = np.random.normal(0, 0.03)
            dy = np.random.normal(0, 0.03)
            for i in range(N):
                new_centers[i][0] += dx
                new_centers[i][1] += dy
                
        if np.random.rand() < 0.05:
            drot = np.random.normal(0, 1.0 * (T / 0.3))
            for i in range(N):
                new_angles[i] += drot

        # Check validity (overlaps)
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            T *= cooling_rate
            continue
            
        # Calculate new cost
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            centers = new_centers
            angles = new_angles
            current_side = new_side
            current_center = new_center
            current_angle = new_angle
            
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in centers]
                best_angles = list(angles)
                best_center = current_center
                best_angle = current_angle
        else:
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                centers = new_centers
                angles = new_angles
                current_side = new_side
                current_center = new_center
                current_angle = new_angle
                
        T *= cooling_rate
        
    return best_centers, best_angles, best_side, best_center, best_angle

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate diverse initial configurations
    initial_configs = _generate_initial_configurations()
    
    best_init_centers = None
    best_init_angles = None
    best_init_side = float('inf')
    
    # Evaluate each initial configuration
    for centers, angles in initial_configs:
        if time.time() - start_time > 15:
            break
        side, result = _try_arrangement(centers, angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result

    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        best_init_centers = fb_centers
        best_init_angles = fb_angles
        best_init_side = fb_side

    # Force-directed relaxation from best initial
    if best_init_centers and time.time() - start_time < DEADLINE - 40:
        est_side, est_center, est_angle = _optimize_outer(best_init_centers, best_init_angles)
        
        relaxed_centers, relaxed_angles = _contact_graph_optimize(
            best_init_centers, best_init_angles, est_side, est_center, est_angle, max_iter=150
        )
        
        if time.time() - start_time < DEADLINE - 35:
            r_side, r_result = _try_arrangement(relaxed_centers, relaxed_angles)
            if r_side < best_init_side:
                best_init_side = r_side
                best_init_centers = [list(c) for c in relaxed_centers]
                best_init_angles = list(relaxed_angles)
                best_result = r_result

    # Run Simulated Annealing
    sa_budget = DEADLINE - time.time() - 30
    if best_init_centers and sa_budget > 5:
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
            best_init_centers, best_init_angles, max_time=sa_budget
        )
        
        if _check_validity(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)

    # Final polish pass with high precision
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        p_centers, p_angles, p_center, p_side, p_angle = best_result
        
        # Use the new gradient polish
        polished_centers, polished_angles = _gradient_polish(
            p_centers, p_angles, p_angle, max_time=DEADLINE - time.time() - 3
        )
        
        if time.time() - start_time < DEADLINE - 1:
            pol_side, pol_result = _try_arrangement(polished_centers, polished_angles, fine=True)
            if pol_side < best_side and pol_result is not None:
                best_side = pol_side
                best_result = pol_result

    # Final verification and formatting
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
