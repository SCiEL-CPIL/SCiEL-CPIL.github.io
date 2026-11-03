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


def smooth_objective_and_grad(x, beta=50.0):
    """
    Smooth objective for L-BFGS-B.
    
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
            if x**2 + y**2 < 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= N:
        # Select N points closest to center, then shuffle
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
    
    return selected


def initial_rings():
    """Generate initial points using concentric rings."""
    points = np.zeros((N, 2))
    
    # Center point
    points[0] = [0, 0]
    count = 1
    
    ring_radii = [0.15, 0.35, 0.55, 0.75, 0.90]
    ring_counts = [1, 6, 12, 18, 13]  # total = 50
    
    for r, n in zip(ring_radii, ring_counts):
        for k in range(n):
            if count >= N:
                break
            angle = 2 * np.pi * k / n
            points[count, 0] = r * np.cos(angle)
            points[count, 1] = r * np.sin(angle)
            count += 1
    
    # Add small random perturbation
    points += np.random.normal(0, 0.02, size=(N, 2))
    
    # Ensure inside unit circle
    norms = np.linalg.norm(points, axis=1)
    for i in range(N):
        if norms[i] > 0.98:
            points[i] *= 0.98 / norms[i]
    
    return points


def initial_random():
    """Generate random initial points inside the unit circle."""
    points = []
    while len(points) < N:
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 < 0.9:
            points.append([x, y])
    return np.array(points)


def lbfbsb_optimize(x0, time_limit=30.0, beta=50.0, maxiter=200):
    """Run L-BFGS-B optimization on the smooth objective."""
    start_time = time.time()
    
    def fun_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        obj, grad = smooth_objective_and_grad(x, beta=beta)
        return obj, grad
    
    try:
        result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                         options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x
    except Exception:
        pass
    
    return x0


def push_apart(points, target_r, iterations=100):
    """Push overlapping circles apart iteratively."""
    points = points.copy()
    
    for _ in range(iterations):
        moved = False
        for i in range(N):
            # Boundary check
            norm = np.linalg.norm(points[i])
            if norm > 1.0 - target_r:
                points[i] = points[i] * (1.0 - target_r) / norm
                moved = True
            
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                min_dist = 2 * target_r
                if dist < min_dist and dist > 1e-10:
                    push = (min_dist - dist) / 2.0
                    direction = diff / dist
                    points[i] += direction * push
                    points[j] -= direction * push
                    moved = True
                elif dist < 1e-10:
                    # Coincident
                    points[i] += np.random.normal(0, 0.01, 2)
                    points[j] -= np.random.normal(0, 0.01, 2)
                    moved = True
        
        if not moved:
            break
    
    # Final boundary enforcement
    for i in range(N):
        norm = np.linalg.norm(points[i])
        if norm > 1.0 - target_r:
            points[i] = points[i] * (1.0 - target_r) / norm
    
    return points


def construct():
    deadline = time.time() + DEADLINE
    
    np.random.seed(42)
    
    best_r = 0.0
    best_points = None
    
    # Generate multiple initializations
    initializations = [
        initial_hexagonal(),
        initial_rings(),
        initial_random(),
    ]
    
    # For each initialization, run L-BFGS-B with increasing beta
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        current_pts = init_pts.copy()
        current_r = compute_radius(current_pts)
        
        if current_r > best_r:
            best_r = current_r
            best_points = current_pts.copy()
        
        # Run L-BFGS-B with different beta values
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 15:
                break
            
            time_left = deadline - time.time() - 10
            if time_left < 5:
                break
            
            x0 = current_pts.flatten()
            optimized = lbfbsb_optimize(x0, time_limit=min(time_left, 25.0), beta=beta, maxiter=150)
            
            opt_pts = optimized.reshape(N, 2)
            opt_r = compute_radius(opt_pts)
            
            if opt_r > current_r:
                current_pts = opt_pts
                current_r = opt_r
            
            if current_r > best_r:
                best_r = current_r
                best_points = current_pts.copy()
        
        # Push-apart refinement
        if time.time() < deadline - 10:
            target_r = max(current_r, 0.12)
            pushed = push_apart(current_pts, target_r, iterations=200)
            pushed_r = compute_radius(pushed)
            
            if pushed_r > current_r:
                current_pts = pushed
                current_r = pushed_r
            
            if current_r > best_r:
                best_r = current_r
                best_points = current_pts.copy()
    
    # Final polish: run L-BFGS-B on the best solution with high beta
    if best_points is not None and time.time() < deadline - 10:
        time_left = deadline - time.time() - 5
        if time_left > 5:
            x0 = best_points.flatten()
            optimized = lbfbsb_optimize(x0, time_limit=min(time_left, 20.0), beta=300.0, maxiter=100)
            opt_pts = optimized.reshape(N, 2)
            opt_r = compute_radius(opt_pts)
            
            if opt_r > best_r:
                best_r = opt_r
                best_points = opt_pts
    
    # Ensure validity
    if best_points is None:
        best_points = initial_hexagonal()
    
    # Final validation and boundary enforcement
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.999:
            best_points[i] *= 0.999 / norms[i]
    
    return best_points


if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
