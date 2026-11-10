"""Pack 300 equal circles in the unit disk.
Strategy: Multi-start global search (hex lattice, sunflower, ring model) with
gradient-based repulsive potential optimization and local refinement using
cKDTree-based force relaxation."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _score(points):
    """Compute the packing score (minimum of pair distances/2 and boundary margins)."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair_dist = dists[:, 1].min() / 2.0
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = (1.0 - radii).min()
    return min(min_pair_dist, min_boundary)


def _repulsive_potential_and_grad(x_flat):
    """
    Computes the value and gradient of the repulsive potential.
    x_flat is a 1D array of length 2*N.
    """
    x = x_flat.reshape(-1, 2)
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on the diagonal (i=j)
    np.fill_diagonal(dist, 1.0)
    
    # Potential: sum_{i<j} 1/dist_ij
    potential = np.sum(1.0 / dist) - N  # Subtract diagonal terms (1/1 = 1)
    
    # Gradient: 
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^3
    dist_cubed = dist ** 3
    np.fill_diagonal(dist_cubed, 1.0)
    
    forces = diff / dist_cubed[:, :, np.newaxis]
    grad_pairwise = -np.sum(forces, axis=1)
    
    # Boundary term: sum_i 1 / (1 - |x_i|)
    r = np.linalg.norm(x, axis=1)
    boundary_potential = np.sum(1.0 / (1.0 - r + 1e-12))
    
    # Gradient of boundary term:
    grad_boundary = np.zeros_like(x)
    mask = r > 1e-6
    grad_boundary[mask] = (1.0 / (1.0 - r[mask]) ** 2) * (x[mask] / r[mask, np.newaxis])
    
    total_potential = potential + boundary_potential
    total_grad = (grad_pairwise + grad_boundary).flatten()
    
    return total_potential, total_grad


def _generate_hex_lattice():
    """Generate hexagonal lattice points inside unit disk."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        points_hex = np.array(pts)
        if len(points_hex) >= N:
            radii = np.hypot(points_hex[:, 0], points_hex[:, 1])
            points_hex = points_hex[np.argsort(radii, kind='stable')[:N]]
            return points_hex
        spacing *= 0.98


def _generate_sunflower():
    """Generate sunflower/phyllotaxis curve points inside unit disk."""
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    r = np.sqrt(i / n) * 0.95
    theta = i * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def _generate_ring_model():
    """Generate ring model initial configuration."""
    # Generate ring counts that sum to N
    counts = [1]
    current = 1
    ring_size = 6
    while current < N:
        add = min(ring_size, N - current)
        if add > 0:
            counts.append(add)
            current += add
        ring_size += 6
    
    # Adjust to sum to N
    diff = N - sum(counts)
    if diff != 0:
        counts[-1] += diff
    
    points_list = []
    total_rings = len(counts)
    for i, n_i in enumerate(counts):
        if n_i == 0:
            continue
        # Radius for this ring
        if i == 0:
            r_i = 0.0
        else:
            r_i = 0.95 * (i / (total_rings - 1)) ** 0.9
        if n_i == 1:
            if r_i < 1e-6:
                points_list.append(np.array([0.0, 0.0]))
            else:
                points_list.append(np.array([r_i, 0.0]))
        else:
            angles = 2.0 * np.pi * np.arange(n_i) / n_i
            x = r_i * np.cos(angles)
            y = r_i * np.sin(angles)
            points_list.append(np.column_stack([x, y]))
    
    points = np.vstack(points_list)
    if len(points) != N:
        # Pad or trim
        if len(points) < N:
            pad = np.random.uniform(-0.5, 0.5, (N - len(points), 2))
            points = np.vstack([points, pad])
        else:
            points = points[:N]
    return points


def _gradient_optimize(points, time_budget=30.0):
    """Optimize using gradient-based method with repulsive potential."""
    start_time = time.time()
    x0 = points.flatten()
    
    def objective(x):
        val, _ = _repulsive_potential_and_grad(x)
        return val
    
    def gradient(x):
        _, grad = _repulsive_potential_and_grad(x)
        return grad
    
    best_points = points.copy()
    best_score = _score(points)
    
    try:
        result = minimize(
            objective, x0, jac=gradient, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.x is not None:
            candidate = result.x.reshape(-1, 2)
            # Ensure inside unit disk
            radii = np.hypot(candidate[:, 0], candidate[:, 1])
            mask = radii >= 1.0
            if mask.any():
                candidate[mask] *= 0.999 / radii[mask]
            candidate_score = _score(candidate)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate
    except Exception:
        pass
    
    return best_points, best_score


def _local_refine(points, time_budget=20.0):
    """Local refinement using force-directed relaxation."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    tree = cKDTree(points)
    
    step_size = 0.005
    min_step = 1e-8
    patience = 30
    no_improve = 0
    start_time = time.time()
    
    k_neighbors = 6
    _, indices = tree.query(points, k=k_neighbors + 1)
    
    while time.time() - start_time < time_budget:
        i = np.random.randint(n)
        
        force = np.zeros(2)
        
        # Boundary force
        r_i = np.hypot(points[i, 0], points[i, 1])
        if r_i > 1e-10:
            boundary_margin = 1.0 - r_i
            if boundary_margin < 0.05:
                force -= points[i] / r_i * (0.05 - boundary_margin) * 2.0
        
        # Neighbor forces
        for j in indices[i, 1:]:
            if j == i:
                continue
            diff = points[i] - points[j]
            d = np.hypot(diff[0], diff[1])
            if d < 2.0 * best_score * 1.5:
                target = 2.0 * best_score
                if d < target:
                    push = (target - d) / max(d, 1e-10)
                    force += diff * push
        
        norm_f = np.hypot(force[0], force[1])
        if norm_f > 1e-12:
            move = force / norm_f * step_size
            new_point = points[i] + move
            
            if np.hypot(new_point[0], new_point[1]) < 1.0:
                old_point = points[i].copy()
                points[i] = new_point
                
                # Quick check
                valid = True
                for j in indices[i, 1:]:
                    if j == i:
                        continue
                    d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
                    if d < 2.0 * best_score * 0.99:
                        valid = False
                        break
                
                if valid:
                    no_improve += 1
                    if no_improve % 20 == 0:
                        new_score = _score(points)
                        if new_score > best_score:
                            best_score = new_score
                            no_improve = 0
                            tree = cKDTree(points)
                            _, indices = tree.query(points, k=k_neighbors + 1)
                        else:
                            points[i] = old_point
                else:
                    points[i] = old_point
                    no_improve += 1
            else:
                no_improve += 1
        
        if no_improve > patience:
            step_size *= 0.8
            no_improve = 0
            if step_size < min_step:
                break
    
    return points, best_score


def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Hexagonal lattice
    try:
        hex_points = _generate_hex_lattice()
        initial_configs.append(('hex', hex_points))
    except Exception:
        pass
    
    # 2. Sunflower curve
    try:
        sun_points = _generate_sunflower()
        initial_configs.append(('sunflower', sun_points))
    except Exception:
        pass
    
    # 3. Ring model
    try:
        ring_points = _generate_ring_model()
        initial_configs.append(('ring', ring_points))
    except Exception:
        pass
    
    if not initial_configs:
        # Fallback: simple random
        points = np.random.uniform(-0.9, 0.9, (N, 2))
        return points
    
    best_score = -np.inf
    best_points = None
    
    # For each initial configuration, run gradient optimization then local refinement
    for name, init_points in initial_configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Gradient optimization
        time_for_grad = min(40, DEADLINE - (time.time() - start_time) - 15)
        if time_for_grad > 5:
            grad_points, grad_score = _gradient_optimize(init_points, time_budget=time_for_grad)
        else:
            grad_points, grad_score = init_points, _score(init_points)
        
        # Local refinement
        time_for_local = min(20, DEADLINE - (time.time() - start_time) - 5)
        if time_for_local > 5:
            final_points, final_score = _local_refine(grad_points, time_budget=time_for_local)
        else:
            final_points, final_score = grad_points, grad_score
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Ensure validity
    if best_points is None:
        best_points = initial_configs[0][1]
    
    best_points = np.array(best_points, dtype=np.float64)
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if (radii >= 1.0).any():
        for i in range(len(best_points)):
            if radii[i] >= 1.0:
                best_points[i] *= 0.999 / radii[i]
    
    return best_points
