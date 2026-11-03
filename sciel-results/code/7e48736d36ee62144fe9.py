import numpy as np
import time

N = 50
DEADLINE = 145.0

def compute_radius(points):
    """
    Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def initial_hexagonal():
    """Generate initial points using a hexagonal lattice pattern."""
    spacing = 0.25
    points = []
    
    max_k = 10
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    
    if len(points) >= N:
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
        
    return selected

def riesz_energy_and_grad(points, s):
    """
    Compute the Riesz kernel energy E = sum_{i<j} |c_i - c_j|^{-s} 
    and its gradient.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of dists[i,j]^-s
    # We can compute sum over all i!=j and divide by 2
    # E = 0.5 * sum_{i!=j} d_ij^-s
    inv_d_s = dists ** (-s)
    np.fill_diagonal(inv_d_s, 0.0)
    energy = 0.5 * np.sum(inv_d_s)
    
    # Gradient: 
    # d/dx_i (d_ij^-s) = -s * d_ij^(-s-1) * (x_i - x_j)
    # grad_i = sum_{j!=i} -s * d_ij^(-s-1) * (x_i - x_j)
    # Note: diff[i,j] = x_i - x_j
    inv_d_s1 = dists ** (-s - 1)
    np.fill_diagonal(inv_d_s1, 0.0)
    
    # grad = -s * sum_j inv_d_s1[i,j] * diff[i,j]
    # inv_d_s1 is (N, N), diff is (N, N, 2)
    grad = -s * np.sum(inv_d_s1[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def gradient_descent_riesz(points, time_limit, s=20.0):
    """
    Minimize Riesz energy using gradient descent with adaptive step size.
    """
    points = points.copy()
    start_time = time.time()
    
    step_size = 0.01
    best_energy = np.inf
    best_points = points.copy()
    
    # We want to minimize energy, which corresponds to maximizing the minimum distance
    # (roughly, for large s, the energy is dominated by the closest pairs)
    
    while time.time() - start_time < time_limit:
        energy, grad = riesz_energy_and_grad(points, s)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Normalize gradient to prevent huge steps
        grad_norm = np.linalg.norm(grad, axis=1)
        max_grad_norm = np.max(grad_norm)
        if max_grad_norm > 1e-9:
            # Scale gradient so that max norm is 1
            scale = 1.0 / max_grad_norm
            grad_scaled = grad * scale[:, np.newaxis]
        else:
            grad_scaled = grad
            
        # Update
        points -= step_size * grad_scaled
        
        # Hard constraint: keep inside circle
        norms = np.linalg.norm(points, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            points[out_mask] = points[out_mask] / norms[out_mask, np.newaxis]
            
        # Adaptive step size: reduce if energy increases significantly
        # (Simplified: just decay step size slowly)
        step_size *= 0.995
        
        # If step size becomes too small, we might be stuck
        if step_size < 1e-6:
            break
            
    return best_points

def construct():
    deadline = time.time() + DEADLINE
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Strategy: Riesz Kernel Gradient Descent
    try:
        init_pts = initial_hexagonal()
        
        # Run gradient descent for a portion of the time
        time_per_run = 5.0
        runs = 0
        while time.time() < deadline - 5.0:
            p = gradient_descent_riesz(init_pts, time_limit=time_per_run, s=20.0)
            r = compute_radius(p)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
            runs += 1
            
            # Perturb initial points for next run to escape local minima
            if runs % 5 == 0:
                # Add some noise to initial points
                noise = np.random.normal(0, 0.05, size=(N, 2))
                init_pts = init_pts + noise
                norms = np.linalg.norm(init_pts, axis=1)
                out_mask = norms > 1.0
                if np.any(out_mask):
                    init_pts[out_mask] = init_pts[out_mask] / norms[out_mask, np.newaxis]
                # Occasionally reset to hexagonal
                if runs % 20 == 0:
                    init_pts = initial_hexagonal()
                    
    except Exception:
        pass
    
    # Final check
    if best_overall_points is not None:
        # One last quick gradient descent to clean up
        if time.time() < deadline - 1:
            p_final = gradient_descent_riesz(best_overall_points, time_limit=1.0, s=20.0)
            r_final = compute_radius(p_final)
            if r_final > best_overall_r:
                best_overall_r = r_final
                best_overall_points = p_final
                
    return best_overall_points if best_overall_points is not None else initial_hexagonal()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
