"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_score = 0.0
    best_points = None
    
    # Helper to calculate the true score (minimum radius)
    def get_score(pts):
        if pts.shape != (N, 2):
            return 0.0
        # Check for non-finite values
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(d_sq)) / 2.0
        # Boundary distance
        radii = np.linalg.norm(pts, axis=1)
        min_boundary = np.min(1.0 - radii)
        return min(min_pair_dist, min_boundary)

    # Smooth energy function for optimization
    # We want to maximize r. 
    # Let's minimize E = sum( (2*r - d_ij)^2 for d_ij < 2*r ) + sum( (1 - |c_i| - r)^2 for |c_i| + r > 1 )
    # But r is a variable. 
    # Alternative: Fix r? No, we want to find max r.
    # Better approach: Minimize a function that is large if constraints are violated.
    # Let's use a barrier-like function.
    # E(pts) = sum_{i<j} max(0, 2*r_est - d_ij)^2 + sum_i max(0, |c_i| + r_est - 1)^2
    # But r_est is tricky. 
    # Let's use the "soft min" idea.
    # Score = min(min_pair/2, min_boundary).
    # Approximate min(a, b) with -log(exp(-beta*a) + exp(-beta*b))/beta.
    # This is smooth.
    
    beta = 100.0 # Sharpness of the soft-min
    
    def smooth_score(pts):
        # Calculate all pairwise distances / 2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq) / 2.0
        
        # Calculate boundary distances
        radii = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - radii
        
        # Flatten
        d_flat = d.flatten()
        # Remove inf
        d_flat = d_flat[np.isfinite(d_flat)]
        
        b_flat = boundary.flatten()
        
        # Combine all constraints into one array
        # We want the minimum of all these values.
        # Values: d_flat (pairwise radii), b_flat (boundary radii)
        all_vals = np.concatenate([d_flat, b_flat])
        
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        # To avoid underflow/overflow, subtract max
        m = np.min(all_vals)
        if m < 0:
             # If any value is negative, the soft min will be dominated by it.
             # Actually, if m is very negative, exp(-beta*m) is huge.
             pass
             
        # Numerical stability
        # Let's just use a large power sum approximation which is easier to code and differentiable
        # min(x) approx - (1/k) * sum(x^k) for large k? No, that's for max.
        # min(x) = -max(-x).
        # max(x) approx (1/k) * sum(x^k) for positive x.
        # Let's stick to log-sum-exp for min.
        
        # exp(-beta * all_vals)
        # If all_vals has large positive numbers, exp(-beta*val) is small.
        # If all_vals has small (negative) numbers, exp(-beta*val) is large.
        # We want to handle the case where values are close to the true min.
        
        # A simpler smooth approximation for the minimum of a set of numbers x_i:
        # f(x) = -1/beta * log( sum_i exp(-beta * x_i) )
        
        # To prevent overflow in exp(-beta * x), we can shift.
        # But since we are maximizing the minimum, the "bad" values are the small ones.
        # If a value is negative, it's very bad.
        
        # Let's compute the soft min directly.
        # Note: if any x_i is -inf, result is -inf.
        
        # For stability:
        min_val = np.min(all_vals)
        # Shift by min_val so that the smallest term is exp(0)=1
        # sum exp(-beta * (x_i - min_val))
        # = sum exp(-beta * x_i + beta * min_val)
        # = exp(beta * min_val) * sum exp(-beta * x_i)
        # log(sum exp(-beta * x_i)) = beta * min_val + log(sum exp(-beta * (x_i - min_val)))
        
        shifted = all_vals - min_val
        exp_terms = np.exp(-beta * shifted)
        log_sum = np.log(np.sum(exp_terms))
        
        soft_min = min_val - (1.0/beta) * log_sum
        
        return -soft_min # We want to minimize negative score

    def smooth_score_grad(pts):
        # Gradient of -soft_min
        # soft_min = min_val - (1/beta) * log(sum(exp(-beta * x_i)))
        # d(soft_min)/d(pts) = - (1/beta) * (1/S) * sum( -beta * exp(-beta * x_i) * d(x_i)/d(pts) )
        # = (1/S) * sum( exp(-beta * x_i) * d(x_i)/d(pts) )
        # where S = sum(exp(-beta * x_i))
        # Note: d(soft_min)/d(pts) is the gradient of the soft min.
        # We are minimizing -soft_min, so grad = - grad(soft_min)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq) / 2.0
        
        radii = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - radii
        
        d_flat = d.flatten()
        d_flat = d_flat[np.isfinite(d_flat)]
        b_flat = boundary.flatten()
        
        all_vals = np.concatenate([d_flat, b_flat])
        
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-beta * shifted)
        S = np.sum(exp_terms)
        
        # We need d(x_i)/d(pts)
        # This is complex because x_i are derived from pts.
        # Let's use finite differences for the gradient if analytic is too complex?
        # Or derive it.
        
        # Let's just use finite differences for the gradient to be safe and simple, 
        # given N=50 is small.
        # Actually, analytic is better. Let's try to construct it.
        
        # x_i for pairs: d_ij/2. 
        # d(d_ij)/d(pts_k) = (pts_i - pts_j)/d_ij * delta_{ik} - (pts_i - pts_j)/d_ij * delta_{jk}
        # d(d_ij/2)/d(pts_k) = 0.5 * (pts_i - pts_j)/d_ij * (delta_{ik} - delta_{jk})
        
        # x_j for boundary: 1 - |c_j|.
        # d(1 - |c_j|)/d(pts_k) = - pts_j / |c_j| * delta_{jk}
        
        # This is doable but tedious to vectorize correctly for the sum.
        # Given the time limit and N=50, a finite difference gradient might be acceptable 
        # if the optimizer is robust, but L-BFGS-B prefers analytic.
        # Let's implement the analytic gradient carefully.
        
        grad = np.zeros_like(pts)
        
        # Pairwise part
        # d is (N, N)
        # d_flat corresponds to upper triangle? No, flatten is row-major.
        # We need to map indices back.
        # It's easier to compute the gradient contribution directly on the (N,N) matrix.
        
        # Let's re-calculate exp_terms on the (N,N) and (N,) arrays.
        
        # Pairs
        d_matrix = d # (N, N)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # We need to handle the min_val shift carefully.
        # min_val is global min of all d_flat and b_flat.
        
        # Let's compute weights w_i = exp(-beta * (x_i - min_val)) / S
        # Then grad(soft_min) = sum w_i * grad(x_i)
        # grad(-soft_min) = - sum w_i * grad(x_i)
        
        # Compute d_flat and b_flat again to get indices
        # This is inefficient. Let's just use finite differences.
        # N=50, 100 variables. 100 evaluations per gradient step.
        # Each evaluation is O(N^2) = 2500 ops. 250k ops per gradient.
        # L-BFGS-B might take 100 steps. 25M ops. Fast enough.
        
        eps = 1e-7
        grad_fd = np.zeros_like(pts)
        for i in range(N):
            for j in range(2):
                old_val = pts[i, j]
                pts[i, j] = old_val + eps
                f_plus = smooth_score(pts)
                pts[i, j] = old_val - eps
                f_minus = smooth_score(pts)
                pts[i, j] = old_val
                grad_fd[i, j] = (f_plus - f_minus) / (2 * eps)
                
        return grad_fd

    # Initial configuration: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    init_points = None
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_points = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98

    # Multi-start optimization
    # We will try a few different initial points and beta values.
    
    starts = [init_points]
    # Add a few random perturbations
    for _ in range(4):
        perturbed = init_points + np.random.randn(N, 2) * 0.05
        starts.append(perturbed)
    
    # Also try a ring configuration
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    ring_pts = np.c_[np.cos(angles), np.sin(angles)] * 0.8
    starts.append(ring_pts)
    
    for start_pts in starts:
        if time.time() > deadline - 10.0:
            break
            
        # Use L-BFGS-B
        # Bounds: each point must be within [-1, 1] roughly, but strictly inside.
        # We can use bounds [(-1, -1), (1, 1)] and rely on the energy function to push them in.
        # Or tighter bounds: [-0.99, 0.99]
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        options = {
            'maxiter': 200,
            'ftol': 1e-10,
            'gtol': 1e-6,
            'maxls': 20
        }
        
        try:
            res = minimize(
                smooth_score, 
                start_pts, 
                method='L-BFGS-B', 
                jac=smooth_score_grad,
                bounds=bounds,
                options=options
            )
            opt_pts = res.x
            # Ensure validity
            if np.all(np.isfinite(opt_pts)) and np.all(np.linalg.norm(opt_pts, axis=1) <= 1.0):
                score = get_score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
        except Exception:
            continue

    # Final Polish: 
    # If we have a good best_points, do a few more iterations with a higher beta (sharper min)
    # to tighten the solution.
    if best_points is not None:
        # Increase beta for sharper optimization
        global beta
        beta = 500.0
        
        # Re-optimize from best_points
        bounds = [(-1.0, 1.0)] * (2 * N)
        options = {
            'maxiter': 100,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 10
        }
        
        try:
            res = minimize(
                smooth_score, 
                best_points, 
                method='L-BFGS-B', 
                jac=smooth_score_grad,
                bounds=bounds,
                options=options
            )
            opt_pts = res.x
            if np.all(np.isfinite(opt_pts)) and np.all(np.linalg.norm(opt_pts, axis=1) <= 1.0):
                score = get_score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback to initial if something went wrong
        return init_points
        
    return best_points
# EVOLVE-BLOCK-END
