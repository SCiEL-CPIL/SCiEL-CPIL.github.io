import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 15.0
    N = 16

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2 = R. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min_max_obj(x, beta=50.0):
        """
        Smooth objective: minimize -log(softmin(d)) + log(softmax(d))
        where softmin(d) ≈ min(d), softmax(d) ≈ max(d).
        This gives a smooth proxy for minimizing (d_max/d_min)^2.
        """
        pts = x.reshape(N, 2)
        d = pdist(pts)
        # Soft min: exp(-beta * min) approximation via log-sum-exp trick
        # softmin(d) = -1/beta * log(sum(exp(-beta*d)))
        # softmax(d) = 1/beta * log(sum(exp(beta*d)))
        
        # For numerical stability, subtract max for softmax
        d_max_val = np.max(d)
        exp_beta_d = np.exp(beta * (d - d_max_val))
        soft_max = d_max_val + np.log(np.sum(exp_beta_d)) / beta
        
        # For softmin, subtract min
        d_min_val = np.min(d)
        exp_neg_beta_d = np.exp(-beta * (d - d_min_val))
        soft_min = d_min_val - np.log(np.sum(exp_neg_beta_d)) / beta
        
        if soft_min <= 1e-12:
            return 1e10
        if soft_max <= 1e-12:
            return 1e10
        
        # We want to maximize soft_min/soft_max, so minimize log(soft_max/soft_min)
        return np.log(soft_max) - np.log(soft_min)

    def soft_min_max_grad(x, beta=50.0):
        """Gradient of soft_min_max_obj."""
        pts = x.reshape(N, 2)
        d = pdist(pts)
        
        d_max_val = np.max(d)
        exp_beta_d = np.exp(beta * (d - d_max_val))
        sum_exp = np.sum(exp_beta_d)
        weights_max = exp_beta_d / sum_exp  # softmax weights
        
        d_min_val = np.min(d)
        exp_neg_beta_d = np.exp(-beta * (d - d_min_val))
        sum_exp_neg = np.sum(exp_neg_beta_d)
        weights_min = exp_neg_beta_d / sum_exp_neg  # softmin weights
        
        if sum_exp_neg < 1e-30:
            return np.zeros_like(x)
        
        # Gradient of log(soft_max) w.r.t. d
        # d(soft_max)/dd = weights_max
        # Gradient of -log(soft_min) w.r.t. d
        # d(soft_min)/dd = -weights_min
        # So d(-log(soft_min))/dd = weights_min / soft_min
        
        soft_max = d_max_val + np.log(sum_exp) / beta
        soft_min = d_min_val - np.log(sum_exp_neg) / beta
        
        if soft_min <= 1e-12 or soft_max <= 1e-12:
            return np.zeros_like(x)
        
        # Gradient w.r.t. each distance d_k
        grad_d = weights_max / soft_max + weights_min / soft_min
        
        # Now need gradient of d_k w.r.t. points
        # d_k = ||p_i - p_j|| for pair (i,j)
        # d(d_k)/dp_i = (p_i - p_j) / d_k
        # d(d_k)/dp_j = (p_j - p_i) / d_k
        
        grad_pts = np.zeros((N, 2))
        idx = np.triu_indices(N, k=1)
        for k in range(len(idx[0])):
            i, j = idx[0][k], idx[1][k]
            diff = pts[i] - pts[j]
            d_k = np.linalg.norm(diff)
            if d_k < 1e-12:
                continue
            gd = grad_d[k] / d_k
            grad_pts[i] += gd * diff
            grad_pts[j] -= gd * diff
        
        return grad_pts.flatten()

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        """Hexagonal arrangement."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def square_ring_init():
        """Points on boundary of a square with some interior."""
        pts = []
        for i in range(4):
            pts.append([-1 + 2*i/3, 1.0])
        for i in range(1, 4):
            pts.append([1.0, 1 - 2*i/3])
        for i in range(1, 4):
            pts.append([-1 + 2*i/3, -1.0])
        for i in range(1, 3):
            pts.append([-1.0, 1 - 2*i/3])
        pts.append([0, 0])
        pts.append([0.5, 0.5])
        pts.append([-0.5, 0.5])
        pts.append([0.5, -0.5])
        pts = np.array(pts[:16], dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        candidates.append(square_ring_init())
        return candidates

    best_pts = None
    best_ratio_sq = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        r = get_ratio_sq(c)
        if r > best_ratio_sq:
            best_ratio_sq = r
            best_pts = c.copy()

    # L-BFGS-B optimization with multiple beta values and restarts
    def lbfgs_optimize(pts, beta=50.0, time_budget=30.0):
        """Run L-BFGS-B on the soft min/max objective."""
        x0 = pts.flatten()
        t_start = time.time()
        
        def obj(x):
            return soft_min_max_obj(x, beta)
        
        def grad(x):
            return soft_min_max_grad(x, beta)
        
        try:
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                      'maxfun': 2000})
            if time.time() - t_start > time_budget:
                # Use the best point found so far
                pass
            x_opt = result.x.reshape(N, 2)
            # Center the result
            x_opt = x_opt - np.mean(x_opt, axis=0)
            return x_opt, get_ratio_sq(x_opt)
        except Exception:
            return pts, get_ratio_sq(pts)

    # Multi-start L-BFGS-B with different beta values
    time_elapsed = time.time() - start_time
    betas_to_try = [30.0, 50.0, 80.0, 120.0]
    
    for beta in betas_to_try:
        time_elapsed = time.time() - start_time
        remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
        if remaining < 10:
            break
        
        budget_per_beta = min(40.0, remaining / len(betas_to_try))
        if budget_per_beta < 5:
            break
        
        # Try from current best
        lbfgs_pts, lbfgs_r = lbfgs_optimize(best_pts, beta=beta, time_budget=budget_per_beta)
        if lbfgs_r > best_ratio_sq:
            best_pts = lbfgs_pts
            best_ratio_sq = lbfgs_r
        
        # Also try from other good initializations
        for c in generate_candidates():
            time_elapsed = time.time() - start_time
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            if len(c) != 16:
                continue
            c_r = get_ratio_sq(c)
            # Only optimize from promising starts
            if c_r > best_ratio_sq * 0.7:
                lbfgs_pts, lbfgs_r = lbfgs_optimize(c, beta=beta, time_budget=min(15.0, budget_per_beta/2))
                if lbfgs_r > best_ratio_sq:
                    best_pts = lbfgs_pts
                    best_ratio_sq = lbfgs_r

    # Random restarts with L-BFGS-B
    time_elapsed = time.time() - start_time
    remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    if remaining > 20:
        np.random.seed(42)
        n_restarts = min(20, int(remaining / 15))
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Perturb best solution
            scale = 0.3 * np.sqrt(remaining / (20 * 15))
            perturbed = best_pts + np.random.randn(16, 2) * scale
            perturbed = perturbed - np.mean(perturbed, axis=0)
            
            lbfgs_pts, lbfgs_r = lbfgs_optimize(perturbed, beta=50.0, time_budget=min(15.0, remaining/10))
            if lbfgs_r > best_ratio_sq:
                best_pts = lbfgs_pts
                best_ratio_sq = lbfgs_r

    # Final high-precision polish with large beta
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 10:
        lbfgs_pts, lbfgs_r = lbfgs_optimize(best_pts, beta=200.0, 
                                             time_budget=min(60.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if lbfgs_r > best_ratio_sq:
            best_pts = lbfgs_pts
            best_ratio_sq = lbfgs_r

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts
