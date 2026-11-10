# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 with pairwise
    disjoint interiors (touching allowed) to maximize the sum of their side lengths.

    Strategy:
      1. Compute a smooth, differentiable upper-bound objective based on Voronoi
         cell inradius: for each square i, its maximum possible side is limited by
         half the distance to the nearest other square's center (in L-infinity metric)
         and to the boundary. We use a smooth approximation of min to get gradients.
      2. Use differential evolution to find good center configurations.
      3. Polish with L-BFGS-B on a smooth objective.
      4. Compute actual side lengths as the maximum feasible (respecting all pairwise
         constraints) and return.

    Returns:
        centers: (10, 2) array
        sides:   (10,) array
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    def elapsed():
        return time.time() - start_time

    # Smooth min using log-sum-exp (for minimizing negative, we use -logsumexp)
    # For a set of distances d_i, smooth_min ≈ -log(sum(exp(-d_i/eps))/eps) * eps
    # Actually, softmin(x) = -logsumexp(-x/eps)*eps
    def softmin(x, eps=0.01):
        # x: array of positive values
        # Returns soft minimum
        neg = -x / eps
        # Shift for numerical stability
        neg_max = np.max(neg)
        log_sum = neg_max + np.log(np.sum(np.exp(neg - neg_max)))
        return -eps * log_sum

    def compute_side_lengths(centers, sides):
        """
        Given centers and proposed sides, compute the maximum feasible side for each
        square such that no two squares overlap and all stay in [0,1]^2.
        This is a linear feasibility problem for each square given the others, but
        we solve it iteratively.
        """
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides.copy()
        
        # Iterate to find consistent side lengths
        for _ in range(100):
            for i in range(n):
                # Boundary constraints
                max_s_boundary = 2.0 * min(cx[i], cy[i], 1.0 - cx[i], 1.0 - cy[i])
                
                # Pairwise constraints
                max_s_pair = max_s_boundary
                for j in range(n):
                    if i == j:
                        continue
                    # |cx_i - cx_j| >= (s_i + s_j)/2  OR  |cy_i - cy_j| >= (s_i + s_j)/2
                    # To avoid overlap, we need at least one of these to hold.
                    # The constraint is: s_i + s_j <= 2*max(|cx_i-cx_j|, |cy_i-cy_j|)
                    # So s_i <= 2*max(|cx_i-cx_j|, |cy_i-cy_j|) - s_j
                    dist_linf = max(abs(cx[i] - cx[j]), abs(cy[i] - cy[j]))
                    max_s_pair = min(max_s_pair, 2.0 * dist_linf - s[j])
                
                s[i] = max(0.0, max_s_pair)
        
        return s

    def objective_smooth(centers_flat):
        """
        Smooth objective: maximize sum of soft-min of distances.
        We minimize the negative of this.
        """
        centers = centers_flat.reshape(n, 2)
        cx = centers[:, 0]
        cy = centers[:, 1]
        
        total = 0.0
        for i in range(n):
            # Distance to boundaries
            dists = np.array([cx[i], cy[i], 1.0 - cx[i], 1.0 - cy[i]])
            
            # Distance to other squares (L-infinity distance between centers)
            for j in range(n):
                if i == j:
                    continue
                d = max(abs(cx[i] - cx[j]), abs(cy[i] - cy[j]))
                dists = np.append(dists, d)
            
            # The side length is limited by 2 * min(dists)
            # We want to maximize sum of 2 * softmin(dists)
            total += 2.0 * softmin(dists, eps=0.02)
        
        return -total  # We minimize negative

    def objective_actual(centers_flat):
        """
        Actual objective: compute feasible side lengths and sum them.
        """
        centers = centers_flat.reshape(n, 2)
        # Initial guess for sides
        sides_guess = np.full(n, 0.3)
        s = compute_side_lengths(centers, sides_guess)
        return -np.sum(s)  # Minimize negative sum

    def is_valid(centers, sides):
        """Check if the configuration is valid."""
        tol = 1e-7
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Check boundaries
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1.0 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1.0 + tol:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        
        # Check finite and positive
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(s)):
            return False
        if not np.all(s > 0):
            return False
        
        return True

    # Initial grid configuration
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_init = (cols + 0.5) * side
    cy_init = (rows + 0.5) * side
    centers_init = np.stack([cx_init, cy_init], axis=1)
    
    best_centers = centers_init.copy()
    best_sides = np.full(n, side)
    best_score = np.sum(best_sides)
    
    # Differential evolution on smooth objective
    bounds = [(0.0, 1.0)] * (2 * n)
    
    try:
        if elapsed() < deadline - 100:
            de_result = differential_evolution(
                objective_smooth,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                callback=lambda x, convergence: (None if elapsed() > deadline - 100 else None)
            )
            
            if elapsed() < deadline - 50:
                centers_de = de_result.x.reshape(n, 2)
                sides_de = compute_side_lengths(centers_de, np.full(n, 0.3))
                if is_valid(centers_de, sides_de):
                    score_de = np.sum(sides_de)
                    if score_de > best_score:
                        best_centers = centers_de
                        best_sides = sides_de
                        best_score = score_de
    except Exception:
        pass

    # Polish with L-BFGS-B on actual objective
    try:
        if elapsed() < deadline - 30:
            result = minimize(
                objective_actual,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if elapsed() < deadline - 10:
                centers_opt = result.x.reshape(n, 2)
                sides_opt = compute_side_lengths(centers_opt, np.full(n, 0.3))
                if is_valid(centers_opt, sides_opt):
                    score_opt = np.sum(sides_opt)
                    if score_opt > best_score:
                        best_centers = centers_opt
                        best_sides = sides_opt
                        best_score = score_opt
    except Exception:
        pass

    # Final validation and return
    if not is_valid(best_centers, best_sides):
        # Fall back to grid
        best_centers = centers_init
        best_sides = np.full(n, side)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
