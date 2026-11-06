import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 8
    
    def compute_score_from_points(pts):
        """
        Compute the score Delta/n^n.
        Delta is product of ordered pair distances after scaling to diameter 2.
        Score = (Prod d_ij) / (Diam^56) * (2^56 / 8^8) ?? 
        The task says: "The score is Delta/n^n, where Delta is the product over ordered pairs of distances after scaling the set to diameter 2".
        Let D be the diameter of the original set.
        Scaled set has diameter 2. Scaling factor is 2/D.
        Distance in scaled set d'_ij = d_ij * (2/D).
        There are n*(n-1) = 8*7 = 56 ordered pairs.
        Delta = Prod_{i!=j} d'_ij = Prod_{i!=j} (d_ij * 2/D) = (Prod_{i!=j} d_ij) * (2/D)^56.
        Score = Delta / n^n = Delta / 8^8.
        
        We want to maximize Score.
        Log(Score) = Log(Delta) - Log(8^8)
                   = Log(Prod_{i!=j} d_ij) + 56 * Log(2/D) - 8 * Log(8)
                   = 2 * Log(Prod_{i<j} d_ij) + 56 * Log(2) - 56 * Log(D) - 8 * Log(8)
        
        Constants: 56 * Log(2) - 8 * Log(8) = 56 * Log(2) - 24 * Log(2) = 32 * Log(2).
        So we maximize: 2 * Sum_{i<j} Log(d_ij) - 56 * Log(D) + Constant.
        """
        pts = np.array(pts, dtype=float)
        if pts.shape != (n, 2):
            return -np.inf
            
        # Check distinctness
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        
        if np.any(d_sq < 1e-12):
            return -np.inf
            
        diam_sq = np.max(d_sq)
        if diam_sq < 1e-12:
            return -np.inf
            
        # Sum of log distances for i < j
        # d_sq contains both i,j and j,i. 
        # Log(Prod_{i!=j} d_ij) = Sum_{i!=j} Log(d_ij) = 2 * Sum_{i<j} Log(d_ij)
        # Sum_{i<j} Log(d_ij) = 0.5 * Sum_{i<j} Log(d_ij^2)
        # Sum_{i!=j} Log(d_ij) = Sum_{i<j} Log(d_ij^2)
        
        log_d_sq_sum_ordered = np.sum(np.log(d_sq))
        
        # Log(Score) proportional to:
        # log_d_sq_sum_ordered - 56 * Log(Diam)
        # Diam = sqrt(diam_sq)
        # Log(Diam) = 0.5 * Log(diam_sq)
        
        log_score_unscaled = log_d_sq_sum_ordered - 56.0 * (0.5 * np.log(diam_sq))
        
        return log_score_unscaled

    def angles_to_points(gaps):
        """
        Convert 7 gaps to 8 points on unit circle.
        gaps: array of 7 floats, positive.
        """
        g = np.array(gaps)
        if np.any(g < 1e-6):
            return None
        total = np.sum(g)
        if total < 1e-6:
            return None
        # Normalize to sum to 2*pi? No, the objective function is scale invariant for angles?
        # If we scale angles, the points move on the circle, but the relative positions change.
        # The gaps define the shape. The sum of gaps must be 2*pi for a closed loop on the circle.
        # So we enforce sum(gaps) = 2*pi by normalizing.
        g_norm = g / total * (2 * np.pi)
        
        angles = np.zeros(n)
        angles[0] = 0
        for i in range(1, n):
            angles[i] = angles[i-1] + g_norm[i-1]
            
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    def objective_from_gaps(gaps):
        pts = angles_to_points(gaps)
        if pts is None:
            return 1e10
        score = compute_score_from_points(pts)
        if score <= -1e9:
            return 1e10
        return -score

    best_score = -np.inf
    best_pts = None
    
    # Regular octagon gaps: pi/4
    regular_gaps = np.array([np.pi/4] * 7)
    regular_pts = angles_to_points(regular_gaps)
    if regular_pts is not None:
        best_score = compute_score_from_points(regular_pts)
        best_pts = regular_pts

    # Strategy 1: Optimize angular gaps
    # Use multiple random starts with Dirichlet distribution
    np.random.seed(42)
    
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate random gaps
        # Dirichlet(1) gives uniform distribution on the simplex
        # We want gaps to be positive and sum to 2*pi.
        # Generate 7 random numbers, normalize.
        raw_gaps = np.random.dirichlet(np.ones(7))
        # Scale to 2*pi
        gaps_init = raw_gaps * (2 * np.pi)
        
        # Also try perturbations of the regular octagon
        if i % 5 == 0:
            perturb = np.random.normal(0, 0.2, 7)
            gaps_init = regular_gaps + perturb
            # Ensure positivity
            gaps_init = np.abs(gaps_init) + 1e-3
            gaps_init = gaps_init / np.sum(gaps_init) * (2 * np.pi)

        try:
            res = minimize(objective_from_gaps, gaps_init, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < 1e9:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    pts_candidate = angles_to_points(res.x)
                    if pts_candidate is not None:
                        best_pts = pts_candidate
        except Exception:
            pass

    # Strategy 2: Full 16-parameter optimization from the best angular solution
    # This allows breaking the "on the circle" constraint if the optimum is not on a circle
    # (Though for max product of distances, convex hull vertices are optimal, and for n=8, 
    # it's highly likely they lie on a circle or close to it).
    # But let's try to refine the coordinates directly.
    
    if best_pts is not None and time.time() < deadline - 20:
        # Scale best_pts to have diameter 1 for numerical stability in optimization?
        # No, the objective is scale invariant.
        
        def full_objective(x):
            pts = x.reshape(n, 2)
            score = compute_score_from_points(pts)
            if score <= -1e9:
                return 1e10
            return -score

        x0 = best_pts.flatten()
        
        try:
            res_full = minimize(full_objective, x0, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res_full.fun < 1e9:
                score_full = -res_full.fun
                if score_full > best_score:
                    best_score = score_full
                    best_pts = res_full.x.reshape(n, 2)
        except Exception:
            pass
        
        # Try SLSQP as well
        try:
            res_slsqp = minimize(full_objective, x0, method='SLSQP',
                                 options={'maxiter': 500, 'ftol': 1e-12})
            if res_slsqp.fun < 1e9:
                score_slsqp = -res_slsqp.fun
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_pts = res_slsqp.x.reshape(n, 2)
        except Exception:
            pass

    # Fallback if something went wrong
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    return best_pts.tolist()

