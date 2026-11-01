"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d_ij) and gradient.
    Energy = -sum_{i<j} log(d_ij)
    grad_i = sum_{j != i} (x_i - x_j) / d_ij^2  (gradient of -log(d) wrt x_i)
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for point i, grad = sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: d_ij^2 = sum(diff^2)
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    
    # grad[i] = sum_j (pts[i] - pts[j]) / dist2[i,j]
    # diff[i,j] = pts[i] - pts[j]
    # So grad[i] = sum_j diff[i,j] / dist2[i,j]
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist2[i,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _halton_init(n, base1=2, base2=3, base3=5):
    """Generate n points using 3D Halton sequence mapped to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base1)
        u2 = halton(i, base2)
        u3 = halton(i, base3)
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts)


def _farthest_point_init(n, start=None):
    """Initialize using farthest-point sampling."""
    if start is None:
        # Start with a random point
        rng = np.random.RandomState(42)
        start = rng.randn(3)
        start = start / np.linalg.norm(start)
    
    pts = [start]
    for _ in range(n - 1):
        # Compute distances from all existing points to all candidate points
        # Use a grid of candidate points
        # For efficiency, use a moderate grid
        theta_vals = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        phi_vals = np.linspace(0, np.pi, 50)
        
        best_dist = -1
        best_point = None
        
        for theta in theta_vals:
            for phi in phi_vals:
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                p = np.array([x, y, z])
                
                # Distance to nearest existing point
                dists = np.linalg.norm(np.array(pts) - p, axis=1)
                min_dist = np.min(dists)
                
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_point = p
        
        pts.append(best_point)
    
    return np.array(pts)


def _lbfgs_optimize(pts, max_iter=500, m=15, deadline=None):
    """L-BFGS optimization with proper line search on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    
    # L-BFGS memory
    s_list = []  # step directions
    y_list = []  # gradient differences
    m_max = m
    
    # Current point and gradient
    x = pts.copy()
    f, g = _energy_and_grad(x)
    best_f = f
    best_x = x.copy()
    
    # Initial step size
    alpha = 0.5
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        if len(s_list) == 0:
            # First iteration: steepest descent
            d = -g
        else:
            # Two-loop L-BFGS recursion
            r = -g
            q = r.copy()
            alpha_coeffs = []
            
            for i in range(len(s_list) - 1, -1, -1):
                s = s_list[i]
                y = y_list[i]
                sy = np.dot(s, y)
                if sy > 1e-12:
                    ai = np.dot(s, q) / sy
                    alpha_coeffs.append(ai)
                    q = q - ai * y
                else:
                    alpha_coeffs.append(0.0)
                    q = q
            
            # Scale
            if len(s_list) > 0:
                s_last = s_list[-1]
                y_last = y_list[-1]
                sy = np.dot(s_last, y_last)
                yy = np.dot(y_last, y_last)
                if sy > 1e-12 and yy > 1e-12:
                    gamma = sy / yy
                else:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            q = gamma * q
            d = q.copy()
            
            for i in range(len(s_list)):
                s = s_list[i]
                y = y_list[i]
                sy = np.dot(s, y)
                if sy > 1e-12:
                    bi = np.dot(y, d) / sy
                    ai = alpha_coeffs[len(s_list) - 1 - i]
                    d = d + (ai - bi) * s
                else:
                    ai = alpha_coeffs[len(s_list) - 1 - i]
                    d = d + ai * s
        
        # Project direction to tangent space
        d = d - np.dot(d, x) * x
        
        # Normalize direction
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            # Gradient is essentially zero in tangent space
            break
        d = d / d_norm
        
        # Line search with Armijo condition
        c1 = 1e-4
        rho = 0.5
        alpha_ls = alpha
        max_ls = 20
        found = False
        
        for ls in range(max_ls):
            # Move along direction and project to sphere
            x_new = _project_to_sphere(x + alpha_ls * d)
            
            # Check minimum distance
            min_dist = 1e9
            for i in range(n):
                for j in range(i + 1, n):
                    d_ij = np.linalg.norm(x_new[i] - x_new[j])
                    if d_ij < min_dist:
                        min_dist = d_ij
            if min_dist < 1e-9:
                alpha_ls *= rho
                continue
            
            f_new, g_new = _energy_and_grad(x_new)
            
            # Armijo condition
            if f_new <= f + c1 * alpha_ls * np.dot(g, d):
                found = True
                break
            
            alpha_ls *= rho
        
        if not found:
            # Try a smaller step
            alpha_ls = alpha * 0.01
            x_new = _project_to_sphere(x + alpha_ls * d)
            f_new, g_new = _energy_and_grad(x_new)
            
            # Check if this is valid
            min_dist = 1e9
            for i in range(n):
                for j in range(i + 1, n):
                    d_ij = np.linalg.norm(x_new[i] - x_new[j])
                    if d_ij < min_dist:
                        min_dist = d_ij
            if min_dist < 1e-9:
                break
        
        # Update L-BFGS memory
        s = alpha_ls * d
        y = g_new - g
        sy = np.dot(s, y)
        
        if sy > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m_max:
                s_list.pop(0)
                y_list.pop(0)
        
        # Update
        x = x_new
        g = g_new
        f = f_new
        
        if f < best_f:
            best_f = f
            best_x = x.copy()
        
        # Adaptive step size
        if f_new < f:
            alpha = min(alpha * 1.2, 2.0)
        else:
            alpha = max(alpha * 0.5, 0.01)
        
        # Convergence check
        if np.linalg.norm(g) < 1e-10:
            break
    
    return best_x, best_f


def _local_search(pts, max_iter=200, deadline=None):
    """Local search with coordinate-wise optimization."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f, _ = _energy_and_grad(pts)
    best_f = f
    best_x = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try moving point i in random directions
            for trial in range(10):
                # Random perturbation
                delta = np.random.randn(3) * 0.1
                p_new = _project_to_sphere(pts[i] + delta)
                
                # Check distance
                min_d = 1e9
                for j in range(n):
                    if j != i:
                        d = np.linalg.norm(p_new - pts[j])
                        if d < min_d:
                            min_d = d
                if min_d < 1e-9:
                    continue
                
                # Evaluate energy change (approximate by recomputing)
                pts_old = pts[i].copy()
                pts[i] = p_new
                f_new, _ = _energy_and_grad(pts)
                pts[i] = pts_old
                
                if f_new < f - 1e-12:
                    pts[i] = p_new
                    f = f_new
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = pts.copy()
        
        if not improved:
            break
    
    return best_x, best_f


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Multiple initialization strategies
    best_x = None
    best_f = np.inf
    
    # Strategy 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Strategy 2: Halton sequence
    pts_halton = _halton_init(N)
    
    # Strategy 3: Farthest point sampling (slower but good diversity)
    if time.time() < deadline - 60:
        pts_fps = _farthest_point_init(N)
    else:
        pts_fps = pts_fib
    
    inits = [pts_fib, pts_halton, pts_fps]
    
    for pts_init in inits:
        if time.time() > deadline - 5:
            break
        
        # L-BFGS optimization
        x_opt, f_opt = _lbfgs_optimize(pts_init, max_iter=500, m=15, deadline=deadline - 10)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
        
        # Local search polish
        if time.time() < deadline - 5:
            x_ls, f_ls = _local_search(x_opt, max_iter=100, deadline=deadline - 2)
            if f_ls < best_f:
                best_f = f_ls
                best_x = x_ls
    
    # Final verification
    if best_x is None:
        best_x = pts_fib
    
    # Ensure points are on unit sphere and no two are too close
    best_x = _project_to_sphere(best_x)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                # Perturb point j
                delta = np.random.randn(3) * 0.01
                best_x[j] = _project_to_sphere(best_x[j] + delta)
    
    return best_x
# EVOLVE-BLOCK-END
