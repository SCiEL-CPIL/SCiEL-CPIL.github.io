"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def calculate_score(points):
        """Calculate the maximum common radius for a given set of points."""
        if len(points) < 2:
            return 1.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dist_sq))
        
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    def gen_hex_lattice(spacing):
        """Generate hexagonal lattice points inside the unit circle."""
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts) if pts else np.empty((0, 2))

    # 1. Initialization
    spacing = np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N))
    points = gen_hex_lattice(spacing)
    
    while len(points) < N and spacing > 0.01:
        spacing *= 0.95
        points = gen_hex_lattice(spacing)
        
    if len(points) >= N:
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        init_points = points[sorted_idx[:N]]
    else:
        init_points = points
        while len(init_points) < N:
            init_points = np.vstack([init_points, np.random.uniform(-0.9, 0.9, size=(1, 2))])
        init_points = init_points[:N]

    best_points = init_points.copy()
    best_score = calculate_score(init_points)

    # 2. Define energy function for L-BFGS-B
    # We use a smooth surrogate for the max-min problem.
    # Maximize r such that |x_i - x_j| >= 2r and |x_i| + r <= 1.
    # Equivalent to minimizing -r subject to constraints.
    # Alternatively, minimize sum of violations.
    # Let's stick to the repulsion energy which drives points apart and to boundary.
    # To improve upon the simple 1/d^2, we can use a softer potential or 
    # optimize the 'temperature' of the soft-min.
    
    # A better approach: use the 'soft-min' of distances.
    # max_r = min( min_{i<j} d_ij/2, min_i (1 - |x_i|) )
    # log-sum-exp trick for min: min(x) = -log(sum(exp(-x/T))) * T
    # But we want to maximize min, so we minimize -min = -(-log(sum(exp(-x/T))) * T) = log(sum(exp(-x/T))) * T
    # This is smooth.
    
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-9)
        dist = np.sqrt(dist_sq)
        
        # Boundary distances
        r_i = np.linalg.norm(x, axis=1)
        d_b = 1.0 - r_i
        
        # Use soft-min surrogate: E = T * log(sum(exp(-V/T)))
        # where V = [d_ij/2 for i<j, d_b_i for i]
        # This smoothly approximates -min(V)
        
        T = 0.005  # Temperature parameter
        
        # Pairwise terms: V_ij = d_ij / 2
        # We want to compute sum over i<j of exp(-d_ij/(2T))
        # Note: dist is symmetric, so we can use upper triangle
        # But for efficiency in gradient computation, let's work with full matrix and divide by 2
        
        # Soft-min of pairwise distances:
        # f_pair = T * log( sum_{i<j} exp(-d_ij / (2T)) )
        # We approximate sum_{i<j} by (sum_{i!=j})/2
        
        # To avoid overflow, use log-sum-exp trick
        # But first, let's compute the log-sum-exp for pairwise terms
        
        # For numerical stability, we'll compute:
        # log_sum_exp(-V/T) = log(sum(exp(-V/T)))
        # = -V_max/T + log(sum(exp(-(V-V_max)/T)))
        
        # Pairwise: V = d_ij/2, so -V/T = -d_ij/(2T)
        # Let's compute this carefully
        
        # Actually, let's use a simpler approach:
        # Compute the soft-min directly with log-sum-exp
        
        # For pairwise distances:
        # S_pair = sum_{i<j} exp(-d_ij / (2T))
        # We can compute sum_{i!=j} exp(-d_ij/(2T)) / 2
        
        # Compute -d_ij / (2T) for all pairs
        neg_V_pair = -dist / (2.0 * T)
        np.fill_diagonal(neg_V_pair, -1e6)  # Exclude diagonal
        
        # Log-sum-exp for numerical stability
        max_neg_V_pair = np.max(neg_V_pair)
        if max_neg_V_pair > -1e5:  # Avoid issues with very large values
            log_S_pair = max_neg_V_pair + np.log(np.sum(np.exp(neg_V_pair - max_neg_V_pair)))
            E_pair = T * log_S_pair
        else:
            E_pair = 0.0  # All distances are large, contribution is negligible
        
        # Gradient of E_pair w.r.t. x_i
        # dE_pair/dx_i = T * (1/S_pair) * sum_{j!=i} (-1/(2T)) * exp(-d_ij/(2T)) * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # So: dE_pair/dx_i = -(1/(2*S_pair)) * sum_{j!=i} exp(-d_ij/(2T)) * (x_i - x_j)
        
        # Compute weights: w_ij = exp(-d_ij/(2T)) / S_pair
        if max_neg_V_pair > -1e5:
            w_pair = np.exp(neg_V_pair - max_neg_V_pair) / np.sum(np.exp(neg_V_pair - max_neg_V_pair))
            # Note: S_pair = exp(max_neg_V_pair) * sum(exp(neg_V_pair - max_neg_V_pair))
            # So w_ij = exp(-d_ij/(2T)) / S_pair = exp(neg_V_pair_ij - max_neg_V_pair) / sum(exp(neg_V_pair - max_neg_V_pair))
        else:
            w_pair = np.zeros_like(neg_V_pair)
        
        np.fill_diagonal(w_pair, 0.0)
        
        # grad_pair_i = -(1/2) * sum_j w_ij * (x_i - x_j)
        # = -(1/2) * sum_j w_ij * diff[i,j]
        grad_pair = -0.5 * np.sum(w_pair[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary terms: V_b = d_b_i = 1 - |x_i|
        # S_bound = sum_i exp(-d_b_i / T)
        neg_V_bound = -d_b / T
        # Handle case where d_b is very small (point near boundary)
        neg_V_bound = np.where(d_b < 1e-6, -1e6, neg_V_bound)
        
        max_neg_V_bound = np.max(neg_V_bound)
        if max_neg_V_bound > -1e5:
            log_S_bound = max_neg_V_bound + np.log(np.sum(np.exp(neg_V_bound - max_neg_V_bound)))
            E_bound = T * log_S_bound
        else:
            E_bound = 0.0
        
        # Gradient of E_bound w.r.t. x_i
        # dE_bound/dx_i = T * (1/S_bound) * (-1/T) * exp(-d_b_i/T) * d(d_b_i)/dx_i
        # d(d_b_i)/dx_i = -x_i / |x_i|
        # So: dE_bound/dx_i = (1/S_bound) * exp(-d_b_i/T) * x_i / |x_i|
        
        if max_neg_V_bound > -1e5:
            w_bound = np.exp(neg_V_bound - max_neg_V_bound) / np.sum(np.exp(neg_V_bound - max_neg_V_bound))
        else:
            w_bound = np.zeros_like(neg_V_bound)
        
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        grad_bound = w_bound[:, np.newaxis] * (x / r_safe[:, np.newaxis])
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # 3. Optimization using L-BFGS-B
    try:
        res = minimize(
            energy_and_grad, 
            best_points.flatten(), 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 5000, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        optimized_points = res.x.reshape((N, 2))
        
        if np.all(np.isfinite(optimized_points)):
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_score(optimized_points)
                if score > best_score:
                    best_points = optimized_points
                    best_score = score
    except Exception:
        pass

    # 4. Local Refinement / Polish
    # Perform a more targeted local search to squeeze out more radius.
    # We can try to move pairs of points that are closest together further apart,
    # while respecting the boundary.
    
    if time.time() < deadline - 10.0:
        current_points = best_points.copy()
        current_score = best_score
        
        # Use a small step size and iterate
        step_size = 0.0005
        no_improve_count = 0
        max_no_improve = 50
        
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            if no_improve_count > max_no_improve:
                break
            
            # Find the bottleneck
            diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_pair_dist = np.min(np.sqrt(dist_sq))
            
            r_i = np.linalg.norm(current_points, axis=1)
            min_bound_dist = np.min(1.0 - r_i)
            
            current_r = min(min_pair_dist / 2.0, min_bound_dist)
            
            if min_pair_dist / 2.0 < min_bound_dist:
                # Pair is limiting
                idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                i, j = idx
                if i == j:
                    continue
                direction = current_points[i] - current_points[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                    # Push apart
                    old_i = current_points[i].copy()
                    old_j = current_points[j].copy()
                    current_points[i] += direction * step_size
                    current_points[j] -= direction * step_size
                    
                    new_score = calculate_score(current_points)
                    if new_score > current_score:
                        current_score = new_score
                        best_points = current_points.copy()
                        best_score = current_score
                        no_improve_count = 0
                    else:
                        # Revert
                        current_points[i] = old_i
                        current_points[j] = old_j
                        no_improve_count += 1
            else:
                # Boundary is limiting
                idx = np.argmin(1.0 - r_i)
                point = current_points[idx]
                norm = np.linalg.norm(point)
                if norm > 1e-9:
                    direction = -point / norm
                    old_p = current_points[idx].copy()
                    current_points[idx] += direction * step_size
                    
                    new_score = calculate_score(current_points)
                    if new_score > current_score:
                        current_score = new_score
                        best_points = current_points.copy()
                        best_score = current_score
                        no_improve_count = 0
                    else:
                        current_points[idx] = old_p
                        no_improve_count += 1
                        # Try a smaller step size if stuck
                        if no_improve_count % 10 == 0:
                            step_size *= 0.9

    return best_points
# EVOLVE-BLOCK-END
