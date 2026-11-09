"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def get_score(pts):
        """Exact score calculation: min radius."""
        # Pairwise distances
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Boundary distance
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    def energy_and_grad(pts, beta):
        """
        Soft-min energy and gradient.
        E = -1/beta * log(sum(exp(-beta * d_ij))) for pairs
        E += -1/beta * log(sum(exp(-beta * (1 - r_i)))) for boundary
        
        Minimizing E maximizes the soft-min distance.
        As beta -> inf, this approaches the hard min.
        """
        pts = np.asarray(pts).reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        
        # Boundary distances
        r = np.linalg.norm(pts, axis=1)
        b = 1.0 - r
        
        # Combine all distances into a single vector for vectorized log-sum-exp
        # We want min(d_ij, b_i).
        # Let's treat them separately to avoid constructing a huge vector if possible, 
        # but N=50 is small. 1225 pairs + 50 boundaries = 1275 distances.
        all_dists = np.concatenate([d.ravel(), b])
        
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        # For numerical stability, subtract max
        x_min = np.min(all_dists)
        # If x_min is negative, we have a violation. The soft min will be negative.
        
        # Calculate log-sum-exp term
        # S = sum(exp(-beta * (d - x_min)))
        # log_S = log(sum(exp(-beta * d) * exp(beta * x_min))) = beta * x_min + log(sum(exp(-beta * (d - x_min))))
        shifted = -beta * (all_dists - x_min)
        # Clip to avoid underflow/overflow issues, though exp(-large) -> 0 is fine.
        # If shifted is very large positive (d << x_min), exp might overflow.
        # But d >= x_min, so d - x_min >= 0, so -beta * (d-x_min) <= 0.
        # So exp is always <= 1. No overflow. Underflow is fine.
        
        log_sum = np.log(np.sum(np.exp(shifted)))
        
        # Soft min value
        soft_min_val = -1.0 / beta * (beta * x_min + log_sum)
        
        # Gradient of soft min with respect to pts
        # d(soft_min)/dx = -1/beta * d/dx [ beta * x_min + log(sum(exp(-beta*(d-x_min)))) ]
        # This is effectively a weighted average of the gradients of the individual distances.
        # weights w_i = exp(-beta * (d_i - x_min)) / sum(exp(-beta * (d_j - x_min)))
        
        exp_terms = np.exp(shifted)
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of d_ij wrt p_i: (p_i - p_j) / d_ij
        # Gradient of b_i wrt p_i: -p_i / r_i
        
        grad = np.zeros_like(pts)
        
        # Pairwise part
        # d is (N, N). weights are flattened (N*N,).
        w_pairs = weights[:N*N].reshape(N, N)
        
        # We need sum_{j != i} w_ij * (p_i - p_j) / d_ij
        # Note: d_ii is inf, w_ii is 0 (exp(-inf) = 0).
        
        # Vectorized computation:
        # G_i = sum_j w_ij * (p_i - p_j) / d_ij
        # diff is (N, N, 2). d is (N, N).
        # term_ij = w_ij * diff_ij / d_ij
        # grad_i = sum_j term_ij
        
        # Handle division by zero (d_ii = inf, so 1/d_ii = 0)
        inv_d = np.zeros_like(d)
        mask = d > 0
        inv_d[mask] = 1.0 / d[mask]
        
        # w_pairs * inv_d is (N, N)
        weighted_inv_d = w_pairs * inv_d
        
        # grad = sum_j weighted_inv_d_ij * diff_ij
        # diff is (N, N, 2). weighted_inv_d is (N, N).
        # We want to sum over axis 1 (j).
        # grad[i] = sum_j weighted_inv_d[i, j] * diff[i, j]
        grad_pairs = np.sum(weighted_inv_d[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary part
        w_bounds = weights[N*N:]
        # b_i = 1 - r_i. grad b_i = -p_i / r_i
        # grad_i += w_i * (-p_i / r_i)
        
        # Handle r_i = 0
        inv_r = np.zeros_like(r)
        mask_r = r > 1e-10
        inv_r[mask_r] = 1.0 / r[mask_r]
        
        grad_bounds = -w_bounds[:, np.newaxis] * pts * inv_r[:, np.newaxis]
        
        grad = grad_pairs + grad_bounds
        
        # We are minimizing -soft_min, so energy = -soft_min_val
        # Energy gradient = -grad
        energy = -soft_min_val
        energy_grad = -grad
        
        return energy, energy_grad

    def hexagonal_initial():
        spacing = 0.2
        points = []
        # Generate a large hex grid
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing + 0.5 * spacing * (j % 2)
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append([x, y])
        points = np.array(points)
        # Filter to inside unit circle (with margin)
        r = np.linalg.norm(points, axis=1)
        points = points[r < 1.0]
        
        # If we have more than N, take the N closest to center? Or random?
        # Taking closest to center creates a dense core.
        if len(points) > N:
            # Sort by distance to center
            r = np.linalg.norm(points, axis=1)
            idx = np.argsort(r)
            points = points[idx[:N]]
        else:
            # If not enough, pad with random points?
            # For N=50 and spacing 0.2, we should have plenty.
            # Area ~ pi. Hex area per point ~ 0.2^2 * sqrt(3)/2 ~ 0.034. 31/0.034 ~ 900 points.
            pass
            
        return points.copy()

    def random_initial():
        pts = np.random.uniform(-1, 1, (N, 2))
        # Push inside
        r = np.linalg.norm(pts, axis=1)
        mask = r > 1.0
        pts[mask] /= r[mask, np.newaxis] * 1.05
        return pts

    best_points = hexagonal_initial()
    best_score = get_score(best_points)
    
    # Multi-start with annealing
    num_starts = 5
    for s in range(num_starts):
        if time.time() > deadline - 15.0:
            break
            
        if s == 0:
            current_pts = hexagonal_initial()
        else:
            # Perturb best
            current_pts = best_points + np.random.randn(N, 2) * 0.1
            # Ensure inside
            r = np.linalg.norm(current_pts, axis=1)
            mask = r > 0.99
            if np.any(mask):
                current_pts[mask] *= 0.99 / r[mask, np.newaxis]

        # Annealing schedule for beta
        # Start with low beta (smooth) and increase
        betas = [50, 100, 200, 500, 1000, 2000]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Use L-BFGS-B
            try:
                res = minimize(
                    lambda x: energy_and_grad(x, beta)[0],
                    current_pts,
                    jac=lambda x: energy_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success:
                    current_pts = res.x
            except Exception:
                pass
            
            # Check score
            curr_score = get_score(current_pts)
            if curr_score > best_score:
                best_score = curr_score
                best_points = current_pts.copy()
                
    # Final polish: Local search on exact score using small steps?
    # Or just run one more L-BFGS with high beta on the best point.
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                lambda x: energy_and_grad(x, 5000)[0],
                best_points,
                jac=lambda x: energy_and_grad(x, 5000)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success:
                final_score = get_score(res.x)
                if final_score > best_score:
                    best_points = res.x
                    best_score = final_score
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
