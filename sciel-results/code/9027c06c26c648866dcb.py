import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    min_bound = 0.5 - np.max(np.abs(positions))
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_bound, min_dist / 2.0)

def soft_min_radius(positions, beta=50.0):
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    Returns a smooth approximation of min(min_bound, min_pair_dist/2).
    """
    # Boundary terms: 0.5 - |x_i|, 0.5 - |y_i|
    bx = 0.5 - np.abs(positions[:, 0])
    by = 0.5 - np.abs(positions[:, 1])
    boundary_terms = np.concatenate([bx, by])
    
    # Pairwise distances
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    all_terms = np.concatenate([boundary_terms, pair_terms])
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * x))) / beta
    m = np.min(all_terms)
    soft_min = m - np.log(np.sum(np.exp(-beta * (all_terms - m)))) / beta
    
    return soft_min

def soft_min_radius_grad(positions, beta=50.0):
    """
    Gradient of the soft-min radius surrogate.
    """
    bx = 0.5 - np.abs(positions[:, 0])
    by = 0.5 - np.abs(positions[:, 1])
    boundary_terms = np.concatenate([bx, by])
    
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    all_terms = np.concatenate([boundary_terms, pair_terms])
    
    m = np.min(all_terms)
    exp_terms = np.exp(-beta * (all_terms - m))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Boundary contributions
    for i in range(N):
        w_bx = weights[i]
        w_by = weights[N + i]
        # d/dx_i of (0.5 - |x_i|) = -sign(x_i)
        grad[i, 0] -= w_bx * np.sign(positions[i, 0])
        grad[i, 1] -= w_by * np.sign(positions[i, 1])
    
    # Pair contributions
    for i in range(N):
        for j in range(i + 1, N):
            w_ij = weights[2*N + (i * (N - 1) - i) + j - i - 1] if False else 0
            # Simpler: iterate over all pairs
            pass
    
    # Let's compute pair weights properly
    pair_weights = weights[2*N:]
    # pair_weights is flattened from upper triangle of dists/2
    # Reconstruct: pair_terms was dists/2 with diagonal inf
    # We need to map back to (i,j) pairs
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Index in pair_terms: row i, col j
            flat_idx = 2 * N + i * N + j
            if flat_idx < len(weights):
                w = weights[flat_idx]
                # d/dx_i of dist(i,j)/2 = (x_i - x_j) / dist(i,j)
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    grad[i] += w * d / dist
                    grad[j] -= w * d / dist
    
    return grad

def generate_hexagonal_initial(N, square_half=0.5):
    """Generate a hexagonal packing initialization."""
    max_coord = 0.5
    spacing = 0.15
    
    points = []
    row = 0
    while True:
        y = -max_coord + row * spacing * np.sqrt(3) / 2
        if y > max_coord:
            break
        x_offset = (row % 2) * spacing / 2
        col = 0
        while True:
            x = -max_coord + col * spacing + x_offset
            if x > max_coord:
                break
            points.append([x, y])
            col += 1
        row += 1
        if len(points) > 200:
            break
            
    points = np.array(points)
    
    if len(points) > N:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    elif len(points) < N:
        return generate_hexagonal_initial(N, square_half, spacing * 0.9)
        
    center = np.mean(points, axis=0)
    points = points - center
    
    max_abs = np.max(np.abs(points))
    if max_abs > 0.48:
        points = points * (0.48 / max_abs)
        
    return points

def force_directed_polish(positions, time_limit=10.0):
    """
    Force-directed method using Delaunay triangulation for dynamic adjacency.
    Applies repulsive forces between Delaunay neighbors to increase minimum distance.
    """
    start_time = time.time()
    best_r = compute_radius(positions)
    best_pos = positions.copy()
    
    positions = positions.copy()
    step = 0.005
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        r = compute_radius(positions)
        if r > best_r + 1e-12:
            best_r = r
            best_pos = positions.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 50:
            step *= 0.95
            no_improve = 0
            if step < 1e-5:
                break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(positions)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
        except Exception:
            edges = set()
            for i in range(N):
                for j in range(i + 1, N):
                    edges.add((i, j))
        
        # Apply forces
        forces = np.zeros((N, 2))
        
        # Repulsion between Delaunay neighbors
        for i, j in edges:
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                # Force proportional to 1/dist^2, pushing apart
                f = d / (dist * dist) * step
                forces[i] += f
                forces[j] -= f
        
        # Boundary forces: push particles away from walls
        for i in range(N):
            for dim in range(2):
                if positions[i, dim] > 0:
                    margin = 0.5 - positions[i, dim]
                    if margin < 0.05:
                        forces[i, dim] -= step * (0.05 - margin) * 10
                else:
                    margin = 0.5 + positions[i, dim]
                    if margin < 0.05:
                        forces[i, dim] += step * (0.05 - margin) * 10
        
        # Apply forces
        positions += forces
        positions = np.clip(positions, -0.4999, 0.4999)
    
    return best_pos

def l_bfgsb_polish(positions, time_limit=15.0):
    """
    L-BFGS-B optimization on the soft-min surrogate objective.
    """
    start_time = time.time()
    best_r = compute_radius(positions)
    best_pos = positions.copy()
    
    beta = 30.0
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 0
        pos = x.reshape(N, 2)
        # Add penalty for going outside bounds
        out_of_bounds = np.maximum(0, np.abs(pos) - 0.5)
        penalty = np.sum(out_of_bounds**2) * 100
        return -soft_min_radius(pos, beta) + penalty
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            return np.zeros_like(x)
        pos = x.reshape(N, 2)
        g = -soft_min_radius_grad(pos, beta)
        # Add gradient of penalty
        out_of_bounds = np.abs(pos) - 0.5
        for i in range(N):
            for dim in range(2):
                if pos[i, dim] > 0.5:
                    g[i, dim] += 200 * (pos[i, dim] - 0.5)
                elif pos[i, dim] < -0.5:
                    g[i, dim] -= 200 * (-0.5 - pos[i, dim])
        return g.flatten()
    
    x0 = positions.flatten()
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 2),
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    if result.success or result.fun < objective(x0):
        new_pos = result.x.reshape(N, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        new_r = compute_radius(new_pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pos
    
    return best_pos

def local_refinement(positions, time_limit=10.0):
    """
    High-precision refinement using iterative constraint satisfaction.
    """
    start_time = time.time()
    best_r = compute_radius(positions)
    best_pos = positions.copy()
    
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    current_step_idx = 0
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        r = compute_radius(positions)
        if r > best_r + 1e-12:
            best_r = r
            best_pos = positions.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 100:
            if current_step_idx < len(step_sizes) - 1:
                current_step_idx += 1
                no_improve = 0
            else:
                break
        
        step = step_sizes[current_step_idx]
        
        min_bound = 0.5 - np.max(np.abs(positions))
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        boundary_margins = 0.5 - np.max(np.abs(positions), axis=1)
        close_to_boundary = boundary_margins < min_bound + 0.003
        for idx in np.where(close_to_boundary)[0]:
            p = positions[idx]
            direction = -p
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[idx] += direction * step * 0.5
        
        close_pairs = dists < min_dist + 0.003
        close_pairs[~np.triu(close_pairs, 1)] = False
        close_idx = np.where(close_pairs)
        
        for i, j in zip(close_idx[0], close_idx[1]):
            p_i = positions[i]
            p_j = positions[j]
            direction = p_i - p_j
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[i] += direction * step * 0.5
                positions[j] -= direction * step * 0.5
                
        positions = np.clip(positions, -0.4999, 0.4999)

    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    positions = generate_hexagonal_initial(N)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.025
    T_min = 1e-7
    cooling_rate = 0.9997
    base_step = 0.035
    
    iter_count = 0
    no_improve_count = 0
    restart_interval = 2500
    
    np.random.seed(42)
    
    # Phase 1: Simulated annealing
    while time.time() < deadline - 40.0:
        iter_count += 1
        
        step_size = base_step * (0.3 + T / 0.025)
        
        idx = np.random.randint(0, N)
        
        delta = np.random.normal(0, step_size, 2)
        old_pos = positions[idx].copy()
        new_pos = old_pos + delta
        
        if np.any(np.abs(new_pos) > 0.5):
            continue
            
        positions[idx] = new_pos
        new_radius = compute_radius(positions)
        
        delta_energy = current_radius - new_radius
        
        if delta_energy < 0:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_positions = positions.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            if np.random.rand() < np.exp(-delta_energy / T):
                current_radius = new_radius
                no_improve_count += 1
            else:
                positions[idx] = old_pos
                no_improve_count += 1
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        if no_improve_count > restart_interval:
            positions = best_positions + np.random.normal(0, 0.012, (N, 2))
            positions = np.clip(positions, -0.495, 0.495)
            current_radius = compute_radius(positions)
            T = 0.012
            no_improve_count = 0
            iter_count = 0
            
    # Phase 2: Force-directed polish with Delaunay
    remaining = deadline - time.time() - 10.0
    if remaining > 8.0:
        best_positions = force_directed_polish(best_positions, time_limit=min(remaining * 0.35, 12.0))
    
    # Phase 3: L-BFGS-B on soft-min surrogate
    remaining = deadline - time.time() - 5.0
    if remaining > 8.0:
        best_positions = l_bfgsb_polish(best_positions, time_limit=min(remaining * 0.5, 15.0))
    
    # Phase 4: High-precision local refinement
    remaining = deadline - time.time() - 2.0
    if remaining > 3.0:
        best_positions = local_refinement(best_positions, time_limit=min(remaining * 0.6, 10.0))
    
    return best_positions
