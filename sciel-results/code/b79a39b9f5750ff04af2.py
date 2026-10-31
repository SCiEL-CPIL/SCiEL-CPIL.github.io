"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def score(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def physics_potential(pts):
        """
        Potential energy function designed to maximize min distance and minimize spread.
        We use a repulsive force F ~ 1/d^2 (potential ~ 1/d) or similar.
        To minimize the ratio max/min, we want points to be as equidistant as possible 
        locally and globally. 
        A common approach for "best packing" or "Thomson problem" is minimizing 
        sum(1/d^p). This pushes close points apart. 
        However, we also need to prevent the cloud from expanding infinitely or collapsing.
        Since the score is scale-invariant, we can fix the scale (e.g., mean distance = 1) 
        or just let the optimizer handle it, but fixing scale helps convergence.
        
        Let's use a potential that penalizes close pairs heavily.
        V = sum_{i<j} (1/d_{ij}^2)
        This minimizes the "crowding". The global minimum of this potential for N points 
        on a plane tends to form a hexagonal lattice or similar close-packed structure 
        if constrained, but in free space it might expand. 
        Actually, for minimizing max/min, we want to equalize distances.
        
        Alternative: Minimize the variance of distances + maximize min distance.
        Let's try a hybrid potential:
        V = alpha * (sum (1/d^2)) + beta * (variance of distances)
        
        Or simpler: Just use the score function directly with a smooth approximation?
        The score (max/min)^2 is non-smooth.
        
        Let's try the "Repulsive" approach first. 
        If we minimize sum(1/d^2), the points will spread out until the repulsion is balanced 
        by... nothing? They will go to infinity. 
        We need an attractive term or a constraint.
        
        Constraint: Fix the diameter (max distance) to 1. Then maximize min distance.
        This is a constrained optimization problem.
        Maximize min(d_{ij}) subject to max(d_{ij}) <= 1.
        
        We can formulate this as:
        Minimize -min(d_{ij}) + penalty * max(0, max(d_{ij}) - 1)^2
        
        Or, since we want to minimize (max/min)^2, let's try to optimize the points 
        using a force-directed layout where:
        1. Pairs with distance < target push apart.
        2. Pairs with distance > target pull together.
        This is essentially trying to make all distances equal to 'target'.
        But there are 120 pairs. We can't make all 120 distances equal in 2D for 16 points.
        
        Let's stick to the "Maximize Min Distance" strategy with a fixed scale.
        1. Normalize points so max_dist = 1.
        2. Objective: Minimize -min_dist.
        3. This is equivalent to finding the best packing of 16 points in a unit circle (roughly).
        
        Let's use scipy.optimize.minimize with Nelder-Mead or L-BFGS-B on a smoothed version.
        Smoothed Min: -logsumexp(-k * d) / k approximates -min(d) as k->inf.
        Let's use k=100.
        
        Objective: f(pts) = -logsumexp(-k * d) / k
        This is smooth and differentiable.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        # Ensure no zero distances
        d = np.maximum(d, 1e-6)
        
        k = 50.0
        # logsumexp(-k*d) / k
        # Use scipy.special.logsumexp for numerical stability? Or just numpy.
        # max_val = np.max(-k * d)
        # val = max_val + np.log(np.sum(np.exp(-k * d - max_val)))
        # result = -val / k
        
        # Simpler: just use -min(d) but it's non-differentiable.
        # Let's try a potential that balances expansion and contraction.
        # V = sum(1/d^2) is good for min dist. 
        # To keep scale, we can add a term that penalizes deviation from a target radius.
        
        # Let's try the "Soft Min" approach.
        max_val = np.max(-k * d)
        lse = max_val + np.log(np.sum(np.exp(-k * d - max_val)))
        return -lse / k

    def physics_potential_grad(pts):
        """
        Gradient of the soft-min potential.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d = np.maximum(d, 1e-6)
        
        k = 50.0
        weights = np.exp(-k * d)
        sum_weights = np.sum(weights)
        
        # Gradient of -logsumexp(-k*d)/k w.r.t d is (1/k) * (sum(w_i * (-k)) / sum(w_i)) = - sum(w_i) / sum(w_i) ? No.
        # d/dd_i [-1/k * log(sum(exp(-k*d_j)))] = -1/k * (exp(-k*d_i) * (-k)) / sum(...) = w_i / sum(w)
        # So gradient w.r.t d_i is w_i / sum(w).
        # Gradient w.r.t pts:
        # d/dx_i (d_{ij}) = (x_i - x_j) / d_{ij}
        
        grad = np.zeros_like(pts)
        # We need to accumulate gradients for all pairs
        # This is O(N^2) per gradient step.
        
        # For each pair i, j:
        # grad_x_i += (w_ij / sum_w) * (x_i - x_j) / d_ij
        # grad_x_j += (w_ij / sum_w) * (x_j - x_i) / d_ij
        
        # Vectorized approach:
        # Create a matrix of weights W where W[i,j] = w_ij / sum_w for i!=j, 0 for i=j
        # Then grad = W @ (pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) / dists
        
        # Let's do it explicitly for clarity and speed with N=16
        for i in range(N):
            for j in range(i+1, N):
                w = weights[j - i - (j - (j-1)//2)] # Indexing in triu is tricky.
                # Let's just recompute d and w in the loop for simplicity, N=16 is small.
                pass
        
        # Better vectorized grad:
        # d_matrix: (N, N)
        # w_matrix: (N, N) with 0 on diagonal
        # w_matrix[i,j] = exp(-k*d_ij) / sum_exp
        # grad_i = sum_j w_matrix[i,j] * (x_i - x_j) / d_ij
        
        d_mat = np.maximum(dists, 1e-6)
        w_mat = np.exp(-k * d_mat)
        np.fill_diagonal(w_mat, 0)
        sum_w = np.sum(w_mat)
        w_mat /= sum_w
        
        # Term (x_i - x_j) / d_ij
        # diff is (N, N, 2)
        # factor is (N, N) where factor[i,j] = 1/d_ij
        factor = 1.0 / d_mat
        np.fill_diagonal(factor, 0)
        
        # grad[i] = sum_j w_mat[i,j] * factor[i,j] * diff[i,j]
        # weighted_diff = w_mat * factor * diff  (N, N, 2)
        weighted_diff = w_mat[:, :, np.newaxis] * factor[:, :, np.newaxis] * diff
        grad = np.sum(weighted_diff, axis=1)
        
        return grad

    # Initial guess: Hexagonal lattice or Grid
    # Hexagonal lattice is better for packing.
    pts = np.zeros((N, 2))
    # Generate hex grid
    side = 4
    count = 0
    for row in range(side):
        for col in range(side):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts[count, 0] = x
            pts[count, 1] = y
            count += 1
            if count == N:
                break
        if count == N:
            break
    
    # Center and scale
    pts -= pts.mean(axis=0)
    max_d = np.max(np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2)))
    pts /= max_d
    
    best_pts = pts.copy()
    best_score = score(pts)
    
    # Optimization Loop
    # We will use L-BFGS-B on the smooth potential.
    # Since the problem is non-convex, we will do multiple restarts.
    
    rng = np.random.default_rng(42)
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            current_pts = best_pts.copy()
        else:
            # Perturb best found
            current_pts = best_pts + rng.normal(0, 0.1, (N, 2))
            current_pts -= current_pts.mean(axis=0)
            max_d = np.max(np.sqrt(np.sum((current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :])**2, axis=2)))
            current_pts /= max_d
        
        # Run local optimization
        # Use L-BFGS-B
        # We minimize the soft-min potential.
        # Note: The potential is scale-invariant if we don't fix scale. 
        # But our potential -logsumexp(-k*d)/k favors larger distances (as d->inf, potential->0).
        # So it will expand the cloud. 
        # We must fix the scale. 
        # Strategy: After each step, re-normalize? 
        # scipy.optimize doesn't support callbacks for normalization easily in the objective.
        # Instead, let's define the objective to include a penalty for deviation from unit scale.
        # Or, simpler: Fix one point at origin and one at (1,0)? No, that breaks symmetry.
        
        # Let's use the "Penalty" method in the objective function for the optimizer.
        def objective(pts_flat):
            pts_2d = pts_flat.reshape(N, 2)
            # Compute distances
            diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            d = np.maximum(d, 1e-6)
            
            # Soft min
            k = 50.0
            max_val = np.max(-k * d)
            lse = max_val + np.log(np.sum(np.exp(-k * d - max_val)))
            soft_min = -lse / k
            
            # Scale penalty: We want max_d to be close to 1.
            max_d = d.max()
            scale_penalty = (max_d - 1.0)**2
            
            # Also penalize if points get too close (to avoid singularity, though soft_min handles it)
            # And maybe variance?
            
            return -soft_min + 10.0 * scale_penalty

        def objective_grad(pts_flat):
            pts_2d = pts_flat.reshape(N, 2)
            diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            d = np.maximum(d, 1e-6)
            
            # Grad of -soft_min
            k = 50.0
            weights = np.exp(-k * d)
            sum_weights = np.sum(weights)
            w_mat = np.zeros((N, N))
            # Fill upper triangle
            w_mat[iu] = weights / sum_weights
            w_mat[(iu[1], iu[0])] = w_mat[iu]
            
            d_mat = np.maximum(dists, 1e-6)
            factor = 1.0 / d_mat
            np.fill_diagonal(factor, 0)
            
            weighted_diff = w_mat[:, :, np.newaxis] * factor[:, :, np.newaxis] * diff
            grad_soft_min = np.sum(weighted_diff, axis=1)
            
            # Grad of scale penalty: 2 * (max_d - 1) * grad(max_d)
            # grad(max_d) is 0 for all pairs except the max pair.
            max_idx = np.unravel_index(np.argmax(dists), dists.shape)
            i, j = max_idx
            if i == j:
                grad_scale = np.zeros_like(pts_2d)
            else:
                # d_ij = ||x_i - x_j||
                # d/dx_i d_ij = (x_i - x_j) / d_ij
                # d/dx_j d_ij = (x_j - x_i) / d_ij
                diff_ij = pts_2d[i] - pts_2d[j]
                d_ij = dists[i, j]
                grad_scale = np.zeros_like(pts_2d)
                grad_scale[i] += 2.0 * (max_d - 1.0) * (diff_ij / d_ij)
                grad_scale[j] += 2.0 * (max_d - 1.0) * (-diff_ij / d_ij)
                
            # Total gradient: grad(-soft_min) + 10 * grad(scale_penalty)
            # Wait, objective is -soft_min + 10*penalty.
            # grad(-soft_min) = - grad(soft_min)
            # My grad_soft_min calculated above was for soft_min (which is negative).
            # Let's check: soft_min = -lse/k. 
            # d(soft_min)/d(pts) = - d(lse)/d(pts) / k
            # d(lse)/d(d_i) = -w_i. 
            # So d(soft_min)/d(d_i) = w_i.
            # My grad_soft_min calculation:
            # weighted_diff = w * (1/d) * diff.
            # This is sum w_ij * (x_i-x_j)/d_ij.
            # This is the gradient of sum w_ij * d_ij? No.
            # d/dx_i (d_ij) = (x_i-x_j)/d_ij.
            # So grad(soft_min) = sum_j w_ij * (x_i-x_j)/d_ij.
            # This matches my code.
            # So grad(-soft_min) = -grad_soft_min.
            
            grad_total = -grad_soft_min + 10.0 * grad_scale
            
            return grad_total.flatten()

        # Run optimization
        try:
            res = minimize(
                objective, 
                current_pts.flatten(), 
                jac=objective_grad, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            opt_pts = res.x.reshape(N, 2)
            
            # Clean up: Center and normalize
            opt_pts -= opt_pts.mean(axis=0)
            
            sc = score(opt_pts)
            if sc < best_score:
                best_score = sc
                best_pts = opt_pts.copy()
                
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
