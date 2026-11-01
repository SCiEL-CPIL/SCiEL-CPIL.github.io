"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin for 120s limit

    # 1. Initial Hexagonal Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    x0 = None
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Ensure points are well inside to allow movement
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            x0 = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98

    def get_radius_candidates(x):
        """
        Returns an array of all 'radius candidates':
        - 1 - |x_i| for each point (distance to boundary)
        - |x_i - x_j| / 2 for each pair (half distance between circles)
        The score is the minimum of this array.
        """
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x) / 2.0
        return np.concatenate([dist_boundary, d_pair])

    def soft_min_score(x, beta):
        """
        Approximates -min(d) using log-sum-exp.
        As beta -> inf, this approaches -min(d).
        We minimize this function.
        """
        x = x.reshape(-1, 2)
        d = get_radius_candidates(x)
        # d should be positive. If any d is negative (overlap/outside), 
        # the soft min will be very large (bad).
        
        # Numerical stability: log(sum(exp(A))) = max(A) + log(sum(exp(A - max(A))))
        # We want to minimize -min(d) ~ (1/beta) * log(sum(exp(-beta * d)))
        A = -beta * d
        max_A = np.max(A)
        # If max_A is very large positive, exp(A - max_A) is safe.
        # If max_A is very large negative, it's also safe.
        log_sum_exp = max_A + np.log(np.sum(np.exp(A - max_A)))
        return log_sum_exp / beta

    def raw_score(x):
        """
        The actual objective to maximize: min(radius_candidates).
        We return negative for minimization.
        """
        x = x.reshape(-1, 2)
        d = get_radius_candidates(x)
        return -np.min(d)

    best_x = x0.copy()
    best_score = -raw_score(best_x)  # raw_score returns negative, so -raw_score is positive min_radius

    bounds = [(-1.0, 1.0)] * (2 * N)

    # 2. Annealing Optimization with Soft-Min
    # Start with smooth (low beta) to allow global rearrangement, then sharpen
    betas = [10, 50, 100, 200, 500]
    
    current_x = best_x.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        def neg_soft_min(x):
            return soft_min_score(x, beta)

        try:
            # Use L-BFGS-B. It's fast and handles bounds.
            # maxiter is kept moderate because we have multiple restarts/betas
            res = minimize(neg_soft_min, current_x, method='L-BFGS-B', 
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < soft_min_score(current_x, beta):
                current_x = res.x.copy()
                
                # Update best raw score
                current_raw = -raw_score(current_x)
                if current_raw > best_score:
                    best_score = current_raw
                    best_x = current_x.copy()
                    
        except Exception:
            continue

    # 3. Final Polish with Raw Objective
    # The soft-min might leave points slightly suboptimal for the strict min.
    # Use a local search on the raw objective.
    if time.time() < deadline - 5:
        # Try Nelder-Mead for a few iterations, it can handle the kinks better than gradient methods
        # but is slower. We limit iterations.
        try:
            res = minimize(raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < -best_score:
                best_x = res.x.copy()
                best_score = -res.fun
        except Exception:
            pass

        # One more L-BFGS-B pass on raw score with tight tolerance
        if time.time() < deadline - 2:
            try:
                res = minimize(raw_score, best_x, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass

    # Final validation: ensure all points are inside the unit circle
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        # Project back inside
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
        
    # Ensure no NaNs or Infs
    if not np.all(np.isfinite(best_x)):
        return x0

    return best_x
# EVOLVE-BLOCK-END
