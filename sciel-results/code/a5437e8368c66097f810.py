import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0


def soft_min_objective_and_grad(x, beta=100.0):
    """
    Smooth objective for L-BFGS-B using log-sum-exp soft-min.
    
    We want to maximize r = min(min_{i<j} d_ij/2, min_i (1-|c_i|)).
    
    Strategy: minimize a smooth surrogate that penalizes:
    1. Boundary violations: for each point, penalty = max(0, |c_i| - 1 + r_est)^2
    2. Overlap violations: for each pair, penalty = max(0, 2*r_est - d_ij)^2
    
    But r_est is unknown. Instead, use a different approach:
    
    Minimize: sum_i max(0, |c_i| - 1)^2 * w_b  +  sum_{i<j} max(0, 2*r - d_ij)^2 * w_p
    
    where r is a running estimate. But this is still non-smooth.
    
    Better: Use a soft-min approximation.
    
    Let f_i = 1 - |c_i|  (boundary margin)
    Let g_ij = d_ij / 2  (pairwise margin)
    
    r = min(f_i, g_ij)
    
    We can approximate -r using: -r ≈ -(1/beta) * log(sum(exp(-beta * f_i)) + sum(exp(-beta * g_ij)))
    
    This is smooth and differentiable. As beta -> inf, it approaches -min(...).
    
    For finite beta, it's a soft version that emphasizes the minimum but is smooth.
    
    Let's use this.
    """
    points = x.reshape(N, 2)
    norms = np.linalg.norm(points, axis=1)
    
    # Boundary margins: f_i = 1 - |c_i|
    # Clip to avoid issues
    f = 1.0 - norms
    
    # Pairwise margins: g_ij = d_ij / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    g = dists / 2.0
    
    # Soft min: -r ≈ -(1/beta) * log(sum(exp(-beta*f)) + sum(exp(-beta*g)))
    # To avoid overflow, use log-sum-exp trick
    
    # Combine all margins into one array
    # f has N elements, g has N*(N-1)/2 elements (upper triangle)
    # We'll work with the full matrix for simplicity, ignoring diagonal
    
    all_margins = np.concatenate([f.ravel(), g.ravel()])
    
    # Log-sum-exp for numerical stability
    max_margin = np.max(all_margins)
    # We want: -(1/beta) * log(sum(exp(-beta * m)))
    # = -(1/beta) * log(sum(exp(beta * (max_margin - m)) * exp(-beta * max_margin)))
    # = -(1/beta) * [log(sum(exp(beta * (max_margin - m)))) + (-beta * max_margin)]
    # = max_margin - (1/beta) * log(sum(exp(beta * (max_margin - m))))
    
    shifted = beta * (max_margin - all_margins)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted))
    
    obj = max_margin - (1.0 / beta) * log_sum_exp
    
    # Gradient
    # d/dx [max_margin - (1/beta) * log(sum(exp(-beta * m_k)))]
    # = d(max_margin)/dx - (1/beta) * d/dx [log(sum(exp(-beta*m_k)))]
    # = d(max_margin)/dx - (1/beta) * [sum_k (-beta * dm_k/dx) * exp(-beta*m_k) / sum_k exp(-beta*m_k)]
    # = d(max_margin)/dx + sum_k [dm_k/dx * exp(-beta*m_k) / sum_k exp(-beta*m_k)]
    
    # But max_margin is the max of all_margins, which is non-differentiable.
    # For the soft-min, the gradient is:
    # grad = sum_k [p_k * grad(m_k)] where p_k = exp(-beta*m_k) / sum_j exp(-beta*m_j)
    # i.e., the gradient is the weighted average of the gradients of each margin,
    # with weights proportional to exp(-beta * m_k).
    
    # p_k = exp(-beta * m_k) / sum_j exp(-beta * m_j)
    # Using the shifted version:
    p = exp_shifted / np.sum(exp_shifted)
    
    # Now compute gradient of each margin w.r.t. x
    grad = np.zeros_like(x)
    
    # Boundary margins: f_i = 1 - |c_i|
    # grad f_i w.r.t. c_i = -c_i / |c_i|  (for |c_i| > 0)
    for i in range(N):
        if norms[i] > 1e-10:
            grad[2*i:2*i+2] += p[i] * (-points[i] / norms[i])
        else:
            # At origin, gradient is undefined; use 0
            grad[2*i:2*i+2] += 0.0
    
    # Pairwise margins: g_ij = d_ij / 2 = |c_i - c_j| / 2
    # grad g_ij w.r.t. c_i = (c_i - c_j) / (2 * d_ij)
    # grad g_ij w.r.t. c_j = (c_j - c_i) / (2 * d_ij)
    
    # p for pairwise: p[N + idx] where idx is the flattened index in the upper triangle
    # But we used the full matrix g (N x N) with diagonal set to inf.
    # Let's re-index: g is N x N, diagonal is inf.
    # all_margins = [f_0, ..., f_{N-1}, g_{00}, g_{01}, ..., g_{N-1,N-1}]
    # But diagonal of g is inf, so exp(-beta*inf) = 0, so they don't contribute.
    
    # Let's be more careful. Let's use the upper triangle only.
    # Actually, let's recompute with explicit upper triangle for clarity.
    
    # Reset and redo gradient more carefully
    grad = np.zeros_like(x)
    
    # Recompute weights
    # f: N values, indices 0..N-1 in all_margins
    # g: N*N values (full matrix), indices N..N+N*N-1 in all_margins
    # Diagonal of g is inf -> weight 0
    
    # Boundary
    for i in range(N):
        w = p[i]
        if norms[i] > 1e-10:
            grad[2*i:2*i+2] += w * (-points[i] / norms[i])
    
    # Pairwise: g_ij for i != j
    # Index in all_margins: N + i*N + j
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = p[N + i*N + j]
            if w < 1e-15:
                continue
            d = dists[i, j]
            if d > 1e-10:
                vec = points[i] - points[j]
                # grad g_ij w.r.t. c_i = vec / (2*d)
                # grad g_ij w.r.t. c_j = -vec / (2*d)
                grad[2*i:2*i+2] += w * (vec / (2.0 * d))
                grad[2*j:2*j+2] += w * (-vec / (2.0 * d))
    
    # The objective is the soft-min value (which approximates r, not -r)
    # Wait: I defined obj = max_margin - (1/beta)*log_sum_exp
    # This approximates min(margin), i.e., r.
    # But we want to MINIMIZE, and we want to MAXIMIZE r.
    # So we should minimize -obj.
    
    # Let me redefine: minimize -r_approx
    # So objective = -obj, gradient = -grad
    
    return -obj, -grad


def initial_hexagonal():
    """Generate initial points using a hexagonal lattice pattern."""
    spacing = 0.25
    points = []
    
    max_k = 8
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


def optimize_with_soft_min(initial_points, time_limit=30.0, beta=100.0):
    """
    Use L-BFGS-B with the soft-min objective to optimize the packing.
    """
    x0 = initial_points.flatten()
    start_time = time.time()
    
    # Define wrapper functions
    def obj_func(x):
        return soft_min_objective_and_grad(x, beta)[0]
    
    def grad_func(x):
        return soft_min_objective_and_grad(x, beta)[1]
    
    # Use L-BFGS-B
    result = minimize(
        fun=obj_func,
        x0=x0,
        jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > time_limit:
        return initial_points, compute_radius(initial_points)
    
    optimized_points = result.x.reshape(N, 2)
    r = compute_radius(optimized_points)
    
    return optimized_points, r


def push_algorithm(points, time_limit=10.0):
    """
    Simple iterative pushing algorithm to resolve overlaps and boundary violations.
    """
    points = points.copy()
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        moved = False
        # Sort by distance to center? Or just iterate.
        for i in range(N):
            # Check boundary
            norm = np.linalg.norm(points[i])
            if norm > 1.0:
                # Push in
                points[i] = points[i] / norm
                moved = True
                
            # Check overlaps
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                # We don't know the exact radius yet, but we can assume a target radius
                # or just push if they are very close.
                # A common heuristic: push if dist < 2 * r_est.
                # Let's estimate r from current best.
                # Or just push if dist < 0.2 (arbitrary threshold).
                # Better: push proportional to overlap.
                
                # Let's assume a target radius r_target = 0.12 (guess).
                # If dist < 2 * r_target, push apart.
                r_target = 0.12
                if dist < 2 * r_target and dist > 1e-8:
                    # Push vector
                    push_vec = diff / dist
                    # Push amount
                    # We want to increase dist to 2 * r_target.
                    # push = (2 * r_target - dist) / 2 for each?
                    # Let's just move them slightly.
                    step = 0.01
                    points[i] += push_vec * step
                    points[j] -= push_vec * step
                    moved = True
                elif dist < 1e-8:
                    # Coincident, push random
                    points[i] += np.random.normal(0, 0.01, 2)
                    points[j] -= np.random.normal(0, 0.01, 2)
                    moved = True
                    
        if not moved:
            break
            
    # Ensure all points are inside
    for i in range(N):
        norm = np.linalg.norm(points[i])
        if norm > 1.0:
            points[i] = points[i] / norm
            
    return points


def construct():
    deadline = time.time() + DEADLINE
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # 1. Hexagonal start + Push + Soft-min optimization
    try:
        init_pts = initial_hexagonal()
        init_pts = push_algorithm(init_pts, time_limit=5.0)
        r_check = compute_radius(init_pts)
        
        if r_check > 0:
            p, r = optimize_with_soft_min(init_pts, time_limit=40.0, beta=100.0)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()
        
    # 2. Ring start + Push + Soft-min optimization
    try:
        points = np.zeros((N, 2))
        points[0] = [0, 0]
        count = 1
        ring_radius = 0.15
        ring_step = 0.15
        
        while count < N:
            n_in_ring = int(2 * np.pi * ring_radius / 0.25)
            if n_in_ring < 1:
                n_in_ring = 1
            
            for k in range(n_in_ring):
                if count >= N:
                    break
                angle = 2 * np.pi * k / n_in_ring
                points[count, 0] = ring_radius * np.cos(angle)
                points[count, 1] = ring_radius * np.sin(angle)
                count += 1
            
            ring_radius += ring_step
            
        points += np.random.normal(0, 0.01, size=(N, 2))
        
        points = push_algorithm(points, time_limit=5.0)
        r_check = compute_radius(points)
        
        if r_check > 0:
            p, r = optimize_with_soft_min(points, time_limit=40.0, beta=100.0)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()

    # 3. Random start + Push + Soft-min optimization
    try:
        if time.time() < deadline - 40:
            pts = []
            while len(pts) < N:
                x, y = np.random.uniform(-1, 1, 2)
                if x**2 + y**2 < 1.0:
                    pts.append([x, y])
                    
            points = np.array(pts)
            points = push_algorithm(points, time_limit=5.0)
            
            r_check = compute_radius(points)
            if r_check > 0:
                p, r = optimize_with_soft_min(points, time_limit=30.0, beta=100.0)
                if r > best_overall_r:
                    best_overall_r = r
                    best_overall_points = p
    except Exception:
        pass

    # Final Polish with soft-min optimization
    if best_overall_points is not None and time.time() < deadline - 5:
        p, r = optimize_with_soft_min(best_overall_points, time_limit=5.0, beta=100.0)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = p
            
    return best_overall_points if best_overall_points is not None else initial_hexagonal()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
