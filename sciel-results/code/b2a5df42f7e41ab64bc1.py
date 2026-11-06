import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # --- Objective and gradient ---
    # Maximize sum of log distances, with soft diameter constraint via log-sum-exp
    
    def compute_dist_matrix(pts):
        # pts: (8, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    
    def log_product_and_grad(pts):
        """
        Returns: log_product (sum of log distances over all ordered pairs i!=j),
                 gradient (8, 2)
        """
        d_mat = compute_dist_matrix(pts)
        mask = ~np.eye(8, dtype=bool)
        
        # Log product: sum over ordered pairs
        d_safe = np.where(mask, np.maximum(d_mat, 1e-15), 1.0)
        log_prod = np.sum(np.log(d_safe) * mask)
        
        # Gradient: d/dp_i sum_j log(|p_i - p_j|) = sum_j (p_i - p_j) / |p_i - p_j|^2
        # For ordered pairs, each pair (i,j) contributes to grad[i] and grad[j]
        # d/dp_i log|p_i - p_j| = (p_i - p_j)/|p_i-p_j|^2
        # d/dp_j log|p_i - p_j| = (p_j - p_i)/|p_i-p_j|^2
        grad = np.zeros_like(pts)
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist2 = np.dot(diff, diff)
                if dist2 < 1e-12:
                    continue
                # Contribution to grad[i]: diff / dist2 (from log|p_i - p_j|)
                # Contribution to grad[j]: -diff / dist2 (from log|p_j - p_i|)
                term = diff / dist2
                grad[i] += term
                grad[j] -= term
        # Since we have ordered pairs (both i->j and j->i in the sum), 
        # actually the sum over ordered pairs means for each unordered pair {i,j},
        # we have log|p_i-p_j| appearing twice (once as (i,j) and once as (j,i)).
        # But wait: "product over ordered pairs" means for each ordered pair (i,j) with i!=j,
        # we include |p_i - p_j|. So there are 8*7 = 56 terms.
        # For each unordered pair {i,j}, |p_i-p_j| appears twice.
        # So log_product = 2 * sum_{i<j} log|p_i-p_j|
        # And the gradient computation above already accounts for both (i,j) and (j,i)
        # because we loop over all i,j with i!=j.
        
        return log_prod, grad
    
    def diameter_soft_constraint(pts, beta):
        """
        Returns: penalty value (0 if all distances <= 2, else soft max exceeding 2)
        and gradient contribution.
        Using: penalty = (1/beta) * log(sum_{i<j} exp(beta * max(0, d_ij - 2)))
        """
        d_mat = compute_dist_matrix(pts)
        # Only consider i < j to avoid double counting
        mask = np.triu(np.ones((8, 8), dtype=bool), k=1)
        
        violations = np.maximum(d_mat - 2.0, 0.0)
        # Soft max via log-sum-exp
        max_viol = np.max(violations * mask)
        if max_viol < 1e-10:
            return 0.0, np.zeros_like(pts)
        
        # log-sum-exp for numerical stability
        shifted = beta * (violations - max_viol) * mask
        # Set non-masked to -inf equivalent
        shifted = np.where(mask, shifted, -1e10)
        log_sum_exp = max_viol + np.log(np.sum(np.exp(shifted)))
        penalty = (1.0 / beta) * log_sum_exp
        
        # Gradient of penalty w.r.t pts
        # d(penalty)/d(d_ij) = (1/beta) * exp(beta*(d_ij - max_viol)) / sum(exp(beta*(d_k - max_viol)))
        # for the violated pairs
        weights = np.where(mask, np.exp(shifted), 0.0)
        weight_sum = np.sum(weights)
        if weight_sum < 1e-15:
            return penalty, np.zeros_like(pts)
        grad_weights = weights / weight_sum  # (1/beta) * w_ij
        
        # Gradient of d_ij w.r.t p_i and p_j
        grad = np.zeros_like(pts)
        for i in range(8):
            for j in range(8):
                if i >= j:
                    continue
                w = grad_weights[i, j]
                if w < 1e-15:
                    continue
                diff = pts[i] - pts[j]
                dist = d_mat[i, j]
                if dist < 1e-10:
                    continue
                # d(d_ij)/d(p_i) = diff/dist, d(d_ij)/d(p_j) = -diff/dist
                term = w * diff / dist
                grad[i] += term
                grad[j] -= term
        
        return penalty, grad
    
    def objective_and_grad(x, beta, lam):
        pts = x.reshape(8, 2)
        log_prod, grad_log = log_product_and_grad(pts)
        pen_val, grad_pen = diameter_soft_constraint(pts, beta)
        
        # We want to maximize log_prod - lam * penalty
        # So minimize -(log_prod - lam * penalty) = -log_prod + lam * penalty
        obj = -log_prod + lam * pen_val
        grad = -grad_log + lam * grad_pen
        
        return obj, grad.flatten()
    
    def score_points(pts):
        """Compute the actual score: Delta / 8^8 where Delta is product over ordered pairs after scaling to diameter 2."""
        # First scale to diameter 2
        d_mat = compute_dist_matrix(pts)
        d_max = np.max(d_mat)
        if d_max < 1e-9:
            return 0.0
        pts_scaled = pts * (2.0 / d_max)
        
        # Product over ordered pairs
        d_mat2 = compute_dist_matrix(pts_scaled)
        mask = ~np.eye(8, dtype=bool)
        d_safe = np.where(mask, np.maximum(d_mat2, 1e-15), 1.0)
        log_delta = np.sum(np.log(d_safe) * mask)
        delta = np.exp(log_delta)
        
        # Score = Delta / n^n = Delta / 8^8
        return delta / (8.0 ** 8)
    
    # --- Optimization with annealing ---
    # Start with regular octagon, then gradually relax symmetry by using multiple starts
    # and annealing the soft constraint
    
    def optimize_from_start(x0, beta_start, lam_start, max_iters, deadline):
        """
        Run L-BFGS-B with adaptive annealing of beta and lambda.
        """
        beta = beta_start
        lam = lam_start
        
        # Initial point
        x = np.array(x0, dtype=float)
        
        # We'll do a sequence of optimizations, decreasing beta (sharper constraint)
        # and increasing lambda (stronger penalty)
        stages = [
            (beta * 1.0, lam * 0.1),
            (beta * 0.5, lam * 0.5),
            (beta * 0.2, lam * 1.0),
            (beta * 0.1, lam * 2.0),
            (beta * 0.05, lam * 4.0),
        ]
        
        best_x = x.copy()
        best_obj = None
        
        for b, l in stages:
            if time.time() > deadline:
                break
                
            def obj_grad_wrapper(xx):
                return objective_and_grad(xx, b, l)
            
            try:
                from scipy.optimize import minimize
                res = minimize(obj_grad_wrapper, x, method='L-BFGS-B',
                              jac=True, options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10})
                x = res.x
            except Exception:
                break
            
            # Evaluate and track best
            obj, _ = objective_and_grad(x, b, l)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_x = x.copy()
        
        return best_x.reshape(8, 2)
    
    # Generate diverse starting points
    def generate_starts():
        starts = []
        
        # 1. Regular octagon
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
        
        # 2. Two concentric squares (various radii and rotations)
        for r_inner in [0.3, 0.5, 0.7]:
            for rot_offset in [0, np.pi/8, np.pi/4]:
                angles_out = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
                angles_in = np.linspace(0, 2*np.pi, 4, endpoint=False) + rot_offset
                p_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
                p_in = np.column_stack([r_inner * np.cos(angles_in), r_inner * np.sin(angles_in)])
                starts.append(np.vstack([p_out, p_in]))
        
        # 3. Perturbed octagon (break symmetry)
        np.random.seed(12345)
        for _ in range(15):
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            radii = 1.0 + np.random.normal(0, 0.3, 8)
            radii = np.maximum(radii, 0.1)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            starts.append(pts)
        
        # 4. Random points in disk
        for _ in range(10):
            r = np.sqrt(np.random.rand(8))
            theta = np.random.rand(8) * 2 * np.pi
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            starts.append(pts)
        
        # 5. Square with corners + midpoints of edges (slightly perturbed)
        for scale in [0.8, 1.0]:
            sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * scale
            mids = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]]) * scale
            pts = np.vstack([sq, mids])
            pts += np.random.normal(0, 0.1, pts.shape)
            starts.append(pts)
        
        return starts
    
    # Main optimization loop
    starts = generate_starts()
    best_score = 0.0
    best_pts = None
    
    # Sort starts by initial score to prioritize good ones
    scored_starts = []
    for s in starts:
        sc = score_points(s)
        scored_starts.append((sc, s))
    scored_starts.sort(key=lambda x: -x[0])
    
    for init_score, start_pts in scored_starts:
        if time.time() > deadline:
            break
        
        try:
            # Run optimization with annealing
            result_pts = optimize_from_start(start_pts, beta_start=5.0, lam_start=10.0, 
                                            max_iters=200, deadline=deadline)
            
            if result_pts is None:
                continue
            
            # Compute score
            sc = score_points(result_pts)
            
            if sc > best_score:
                best_score = sc
                best_pts = result_pts.copy()
                
        except Exception:
            continue
    
    # Fallback to regular octagon if nothing better
    if best_pts is None or best_score < 1.0:
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = 1.0
    
    # Final scaling to ensure diameter exactly 2
    d_mat = compute_dist_matrix(best_pts)
    d_max = np.max(d_mat)
    if d_max > 1e-9:
        best_pts = best_pts * (2.0 / d_max)
    
    # Ensure coordinates within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    return best_pts.tolist()
