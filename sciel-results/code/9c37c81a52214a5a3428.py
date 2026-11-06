"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        if min_boundary < 0:
            return 0.0
        # Use grid-based nearest neighbor for efficiency
        r_est = min_boundary
        # Quick check: if r_est is very small, we can skip
        # Use spatial hash to find min pairwise distance
        cell_size = max(r_est * 2.5, 0.01)
        if cell_size <= 0:
            cell_size = 0.01
        grid = defaultdict(list)
        for i in range(N):
            cx = int(pts[i, 0] / cell_size)
            cy = int(pts[i, 1] / cell_size)
            grid[(cx, cy)].append(i)
        min_dist = np.inf
        for i in range(N):
            cx = int(pts[i, 0] / cell_size)
            cy = int(pts[i, 1] / cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (cx + dx, cy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if j > i:
                                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                                if d < min_dist:
                                    min_dist = d
        r_pair = min_dist / 2.0
        return min(r_pair, min_boundary)

    def generate_hex_lattice(r):
        """Generate hexagonal lattice points fitting in unit circle."""
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        for j in range(-k, k + 1):
            offset = 0.5 * d if (j % 2 != 0) else 0.0
            for i in range(-k, k + 1):
                x = i * d + offset
                y = j * d * sqrt3 / 2.0
                if np.hypot(x, y) + r <= 1.0:
                    points.append([x, y])
        if len(points) == 0:
            return np.array([])
        points = np.array(points)
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        return points

    def optimize(points, max_time=140.0):
        """
        Optimize packing using soft-min gradient with spatial hashing.
        """
        points = points.copy()
        best_points = points.copy()
        best_r = compute_radius(points)
        
        # Soft-min temperature (beta). Higher = more focused on binding constraints.
        beta = 80.0
        step_size = 0.003
        decay = 0.9997
        no_improve_count = 0
        patience = 3000
        
        # Precompute for efficiency
        for it in range(50000):
            if time.time() > max_time:
                break
            if it % 500 == 0 and time.time() > max_time:
                break
                
            # Compute current radius
            current_r = compute_radius(points)
            if current_r > best_r + 1e-12:
                best_r = current_r
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > patience:
                # Perturb to escape local optimum
                num_perturb = N // 8
                idx = np.random.choice(N, num_perturb, replace=False)
                perturbations = np.random.randn(num_perturb, 2) * 0.01
                points[idx] += perturbations
                # Project back inside
                radii = np.hypot(points[:, 0], points[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    points[outside] /= radii[outside, np.newaxis]
                no_improve_count = 0
                continue
            
            # Build spatial hash grid
            # Cell size should be ~2*r_est so we catch all potential neighbors
            r_est = max(best_r, 0.001)
            cell_size = max(2.5 * r_est, 0.01)
            grid = defaultdict(list)
            for i in range(N):
                cx = int(np.floor(points[i, 0] / cell_size))
                cy = int(np.floor(points[i, 1] / cell_size))
                grid[(cx, cy)].append(i)
            
            # Compute soft-min gradient
            # Objective: maximize m = soft_min(f_ij/2, g_i)
            # where f_ij = dist(i,j), g_i = 1 - |p_i|
            # soft_min = -1/beta * log(sum(exp(-beta * x)))
            # gradient wrt x_k = -exp(-beta*x_k) / sum(exp(-beta*x_j))
            # We want to move points to INCREASE m, so move along +gradient of m wrt positions
            
            forces = np.zeros_like(points)
            
            # First pass: compute all constraint values and their exp(-beta * val)
            # We'll accumulate weighted forces
            
            # Boundary constraints: g_i = 1 - |p_i|
            radii = np.hypot(points[:, 0], points[:, 1])
            g_vals = 1.0 - radii  # N values
            # exp(-beta * g_i), but clip to avoid underflow
            beta_g = np.clip(-beta * g_vals, -500, 500)
            exp_g = np.exp(beta_g)
            
            # Pairwise constraints: f_ij = dist(i,j)/2
            # We'll collect these via the grid
            pair_vals = []  # (i, j, val)
            for i in range(N):
                cx = int(np.floor(points[i, 0] / cell_size))
                cy = int(np.floor(points[i, 1] / cell_size))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        key = (cx + dx, cy + dy)
                        if key in grid:
                            for j in grid[key]:
                                if j > i:
                                    d = np.hypot(points[i, 0] - points[j, 0],
                                                  points[i, 1] - points[j, 1])
                                    pair_vals.append((i, j, d / 2.0))
            
            # Collect all constraint values
            # Boundary: N values
            # Pairwise: ~N * avg_neighbors / 2 values
            all_vals = np.concatenate([g_vals, [p[2] for p in pair_vals]])
            
            # Compute soft-min weights
            # w_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
            # To avoid overflow/underflow, use log-sum-exp trick
            max_val = np.max(all_vals)
            shifted = -beta * (all_vals - max_val)
            shifted = np.clip(shifted, -500, 500)
            exp_shifted = np.exp(shifted)
            total_exp = np.sum(exp_shifted)
            if total_exp < 1e-300:
                total_exp = 1e-300
            weights = exp_shifted / total_exp
            
            # First N weights are for boundary, rest for pairs
            boundary_weights = weights[:N]
            pair_weights = weights[N:]
            
            # Boundary force: gradient of g_i = 1 - |p_i| wrt p_i
            # d/dp_i (1 - |p_i|) = -p_i / |p_i|
            # We want to move in direction that INCREASES g_i, so +gradient
            # Force_i_boundary = boundary_weights[i] * (-p_i / |p_i|)
            # But we want to maximize m, and dm/dp_i = sum_k w_k * dx_k/dp_i
            # For boundary: dx_k/dp_i = -p_i/|p_i| for k=i (the i-th boundary term)
            # So force = w_i * (-p_i/|p_i|) ... but we want to move to increase, so:
            # Actually m = soft_min, and we want to increase it.
            # dm/dp_i = sum_k w_k * (dx_k/dp_i)
            # For boundary term k=i: dx_i/dp_i = -p_i/|p_i|
            # So contribution to force from boundary = w_i * (-p_i/|p_i|)
            # We move points in the direction of +dm/dp_i to increase m.
            
            safe_radii = np.where(radii > 1e-8, radii, 1.0)
            boundary_force = -boundary_weights[:, np.newaxis] * points / safe_radii[:, np.newaxis]
            forces += boundary_force
            
            # Pairwise forces: for pair (i,j) with value d_ij/2
            # d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij) ... wait
            # d_ij = |p_i - p_j|, so d(d_ij/2)/dp_i = (p_i - p_j) / (2 * d_ij) * 2 = (p_i - p_j)/d_ij
            # Wait: d/dp_i |p_i - p_j| = (p_i - p_j)/|p_i - p_j|
            # So d/dp_i (d_ij/2) = (p_i - p_j)/(2 * d_ij) ... no
            # d_ij/2, derivative wrt p_i: (1/2) * (p_i - p_j)/d_ij
            # So force contribution to point i from pair (i,j): w_pair * (p_i - p_j)/(2*d_ij)
            # And to point j: w_pair * (p_j - p_i)/(2*d_ij)
            
            for idx, (i, j, val) in enumerate(pair_vals):
                w = pair_weights[idx]
                pi = points[i]
                pj = points[j]
                d = 2.0 * val  # actual distance
                if d < 1e-10:
                    # Nearly coincident, push apart in random direction
                    direction = np.random.randn(2)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    forces[i] += w * direction * 0.01
                    forces[j] -= w * direction * 0.01
                else:
                    diff = pi - pj
                    force_mag = w / (2.0 * d)
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
            
            # Scale and apply
            # Normalize forces to have reasonable magnitude
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                forces = forces / max_norm * 0.5
            
            # Update
            new_points = points + forces * step_size
            
            # Project inside unit circle
            new_radii = np.hypot(new_points[:, 0], new_points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                new_points[outside] = new_points[outside] / new_radii[outside, np.newaxis]
            
            # Accept if it improves or is close
            new_r = compute_radius(new_points)
            if new_r >= current_r - 1e-10:
                points = new_points
            else:
                # Rejected, reduce step
                step_size *= 0.9
            
            # Decay step size
            step_size = max(step_size * decay, 0.0001)
            
            # Occasionally increase beta to focus more
            if it % 2000 == 0:
                beta = min(beta * 1.1, 200.0)
        
        return best_points, best_r

    # 1. Initialization
    # Find good hex lattice starting point
    r_low = 0.01
    r_high = 0.08
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
    
    for _ in range(25):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
    
    points = generate_hex_lattice(r_low)
    if len(points) < N:
        # Fallback: random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.6
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
    
    # 2. Multi-restart optimization
    best_points = points.copy()
    best_r = compute_radius(points)
    
    n_restarts = 5
    time_per_restart = (deadline - time.time()) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            opt_points = points.copy()
        else:
            # Perturb from best
            opt_points = best_points.copy()
            num_perturb = N // 5
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * 0.02
            opt_points[idx] += perturbations
            # Project inside
            radii = np.hypot(opt_points[:, 0], opt_points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                opt_points[outside] /= radii[outside, np.newaxis]
        
        remaining = deadline - 5.0 - time.time()
        if remaining < 2.0:
            break
        
        opt_result, opt_r = optimize(opt_points, max_time=time.time() + remaining)
        
        if opt_r > best_r:
            best_r = opt_r
            best_points = opt_result.copy()
    
    # Final verification
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
    
    return best_points
# EVOLVE-BLOCK-END
