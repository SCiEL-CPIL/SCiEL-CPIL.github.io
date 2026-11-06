import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius for the given points.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def soft_min_radius_and_grad(points, beta):
    """
    Computes the soft-min radius and its gradient.
    """
    points = np.asarray(points)
    N = len(points)
    
    # 1. Compute pairwise distances and their gradients
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Values f_k for pairs
    f_pairs = dists / 2.0
    
    # Values f_k for boundaries
    norms = np.linalg.norm(points, axis=1)
    f_bound = 1.0 - norms
    
    # Combine for finding min
    iu = np.triu_indices(N, k=1)
    f_pairs_vals = f_pairs[iu]
    f_bound_vals = f_bound
    
    all_f = np.concatenate([f_pairs_vals, f_bound_vals])
    m = np.min(all_f)
    
    # Compute shifted exponentials
    w_pairs = np.exp(-beta * (f_pairs - m))
    np.fill_diagonal(w_pairs, 0.0)
    
    S_pairs = np.sum(w_pairs)
    
    # Gradient contribution from pairs
    dists_safe = np.maximum(dists, 1e-12)
    np.fill_diagonal(dists_safe, 1.0)
    
    grad_terms = w_pairs[:, :, np.newaxis] * diff / (2.0 * dists_safe[:, :, np.newaxis])
    grad_pairs = np.sum(grad_terms, axis=1)
    
    # 2. Boundary
    w_bound = np.exp(-beta * (f_bound - m))
    S_bound = np.sum(w_bound)
    
    norms_safe = np.maximum(norms, 1e-12)
    grad_bound = w_bound[:, np.newaxis] * (-points / norms_safe[:, np.newaxis])
    
    S = S_pairs + S_bound
    
    soft_min_val = -m - (1.0 / beta) * np.log(S)
    
    grad_soft_min = (grad_pairs + grad_bound) / S
    
    return soft_min_val, grad_soft_min

def md_optimize(points, time_limit, seed=42):
    """
    Optimizes using Langevin Dynamics (Molecular Dynamics) with annealing.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    pts = points.copy()
    vel = np.zeros_like(pts)
    
    # Parameters
    dt = 0.01
    damping = 0.9
    k_rep = 1.0
    k_center = 0.1 # Initial center attraction, will increase
    
    # Annealing schedule
    # We want to increase the repulsion strength or decrease temperature?
    # Standard: High T -> Low T.
    # Force from repulsion: F ~ 1/d^2.
    # We can scale the forces or add noise.
    
    # Let's use a temperature T that scales the noise.
    # Noise ~ sqrt(T) * random.
    
    T_start = 0.1
    T_end = 0.001
    num_steps = 2000
    
    # We also want to "tighten" the packing.
    # We can increase k_center over time to push points to the boundary?
    # No, maximizing radius means we want them as far apart as possible.
    # If we just repel, they go to the boundary.
    # The constraint is the boundary.
    # So we need a repulsive force from the boundary if they go outside.
    
    # Let's define the potential:
    # U = sum_{i<j} k_rep / d_ij^2 + sum_i k_bound * max(0, |x_i| - (1 - r_est))^2
    # This is tricky because r_est changes.
    
    # Alternative: Just use repulsion between points and a hard wall.
    # If |x_i| > 1 - r_current, push back.
    # But we don't know r_current.
    
    # Simpler: Minimize U = sum_{i<j} 1/d_ij^2 + lambda * sum |x_i|^2.
    # As lambda increases, points are pulled to center, increasing density.
    # We anneal lambda from small to large.
    
    lambda_start = 0.01
    lambda_end = 10.0
    
    for step in range(num_steps):
        if time.time() > time_limit:
            break
            
        # Interpolate lambda and T
        t = step / (num_steps - 1)
        lambda_curr = lambda_start * (lambda_end / lambda_start) ** t
        T_curr = T_start * (T_end / T_start) ** t
        
        # Calculate Forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                d_sq = np.dot(diff, diff)
                if d_sq < 1e-6:
                    # Jitter if too close
                    diff += np.random.normal(0, 0.01, 2)
                    d_sq = np.dot(diff, diff)
                d = np.sqrt(d_sq)
                # F = -grad(1/d^2) = 2/d^3 * diff
                # Scale by k_rep
                f_mag = 2.0 * k_rep / (d_sq * d)
                f_vec = f_mag * diff
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Center attraction
        for i in range(N):
            forces[i] -= lambda_curr * pts[i]
        
        # Update Velocities
        vel = vel * damping + forces * dt + np.random.normal(0, np.sqrt(T_curr * dt), (N, 2))
        
        # Update Positions
        pts = pts + vel * dt
        
        # Hard constraint: Keep inside unit circle (with small margin)
        # If a point goes outside, reflect it or clamp it.
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 0.99: # Margin
                # Project back to circle of radius 0.99
                pts[i] = pts[i] * (0.99 / norms[i])
                # Damp velocity component normal to boundary
                normal = pts[i] / norms[i]
                v_dot_n = np.dot(vel[i], normal)
                if v_dot_n > 0:
                    vel[i] -= 2 * v_dot_n * normal
                    vel[i] *= 0.5 # Damping on reflection
    
    return pts

def polish(points, time_limit):
    """
    Polishes the result using L-BFGS-B on soft-min.
    """
    start_time = time.time()
    current_points = points.copy()
    
    # Use a high beta to approximate the min
    beta = 500.0
    
    def obj(x):
        val, grad = soft_min_radius_and_grad(x.reshape(N, 2), beta)
        return -val, -grad.flatten()
    
    try:
        res = minimize(obj, current_points.flatten(), method='L-BFGS-B', 
                       jac=True, options={'maxiter': 200, 'gtol': 1e-8})
        current_points = res.x.reshape(N, 2)
    except Exception:
        pass
        
    return current_points

def generate_seeds():
    seeds = []
    
    # 1. Hexagonal
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0], radius[radius <= 1.0]
        if len(points) >= N:
            hex_seed = points[np.argsort(radius, kind="stable")[:N]]
            seeds.append(hex_seed)
            break
        spacing *= 0.98
        
    # 2. Random
    np.random.seed(42)
    for i in range(5):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(pts)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    seeds = generate_seeds()
    
    best_points = None
    best_score = -1.0
    
    # Time allocation:
    # MD is slow (O(N^2) per step).
    # Let's give 15s to each MD run.
    # 7 seeds * 15s = 105s.
    # Plus polish time.
    
    md_time_per_seed = 12.0
    polish_time_per_seed = 2.0
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5.0:
            break
            
        # MD Phase
        t_start = time.time()
        md_limit = t_start + md_time_per_seed
        try:
            md_pts = md_optimize(seed, md_limit, seed=42+i)
        except Exception:
            md_pts = seed
            
        # Polish Phase
        t_start = time.time()
        polish_limit = t_start + polish_time_per_seed
        try:
            polished_pts = polish(md_pts, polish_limit)
        except Exception:
            polished_pts = md_pts
            
        score = compute_radius(polished_pts)
        if score > best_score:
            best_score = score
            best_points = polished_pts
            
    # Final polish on best candidate if time remains
    if best_points is not None and time.time() < deadline - 2.0:
        time_left = deadline - time.time() - 1.0
        try:
            # Run a longer polish
            def long_polish(points, limit):
                start = time.time()
                beta = 1000.0
                def obj(x):
                    val, grad = soft_min_radius_and_grad(x.reshape(N, 2), beta)
                    return -val, -grad.flatten()
                res = minimize(obj, points.flatten(), method='L-BFGS-B', jac=True, options={'maxiter': 500, 'gtol': 1e-10})
                return res.x.reshape(N, 2)
            
            final_pts = long_polish(best_points, time_left)
            final_score = compute_radius(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
        except Exception:
            pass
            
    if best_points is None:
        best_points = seeds[0]
        
    return best_points

if __name__ == "__main__":
    points = construct()
    r = compute_radius(points)
    print(f"Radius: {r}")
