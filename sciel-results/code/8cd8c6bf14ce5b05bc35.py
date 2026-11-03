import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 26

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin before 150s kill

    best_score = -np.inf
    best_solution = None

    def _elapsed():
        return time.time() - start_time

    def _valid(z):
        """Check if a full (N,3) solution is valid: no overlaps, inside square."""
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        # Boundary
        if np.any(r < -1e-9) or np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6) or \
           np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
            return False
        # Overlap
        pts = np.column_stack((x, y))
        dist = cdist(pts, pts)
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(d_ij < r_sum - 1e-6):
            return False
        return True

    def _score(z):
        return float(np.sum(z[:, 2]))

    def _radii_from_centers(c):
        """
        Given N centers (N,2), compute radii as a soft-min of:
          - distance to each of the 4 walls
          - distance to each other center / 2 (approximation: min pairwise dist / 2)
        Using a log-sum-exp style soft min for differentiability.
        """
        x = c[:, 0]
        y = c[:, 1]
        # Wall distances for each circle
        wall_dists = np.column_stack((x, 1 - x, y, 1 - y))  # (N, 4)
        # Pairwise center distances
        pts = c
        dist = cdist(pts, pts)  # (N, N)
        # For each circle i, the min distance to any other circle j
        # Set diagonal to inf
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, np.inf)
        min_pair_dist = np.min(dist_safe, axis=1)  # (N,)
        # The "available" radius for circle i is min(wall_dists[i], min_pair_dist[i]/2)
        # But we want a smooth version. Let's combine all constraints per circle.
        # For circle i, constraints are: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i,
        # and r_i <= d_ij/2 for all j != i.
        # So the max possible r_i is min of all these.
        # We use a soft min over all these values.
        
        # Build a matrix of all constraint values per circle: (N, N+3)
        # Columns 0-3: wall distances
        # Columns 4 to N+2: d_ij/2 for j != i (with some padding)
        # Actually, let's just compute the hard min and then smooth it.
        
        # Hard min per circle
        all_vals = np.column_stack([wall_dists, min_pair_dist / 2.0])  # (N, 5)
        # Soft min using log-sum-exp trick: softmin(x) = -logsumexp(-x / t) * t
        # For t -> 0, it approaches min. For t > 0, it's smooth.
        t = 0.01
        # Clip to avoid issues
        all_vals_clipped = np.clip(all_vals, 1e-8, 1.0)
        neg_vals = -all_vals_clipped / t
        # Log-sum-exp for numerical stability
        max_neg = np.max(neg_vals, axis=1, keepdims=True)
        logsumexp = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg), axis=1, keepdims=True))
        soft_min = -logsumexp * t  # (N, 1)
        r = soft_min.flatten()
        # Ensure positive and reasonable
        r = np.clip(r, 1e-6, 0.5)
        return r

    def _objective_and_grad(c):
        """
        Objective: maximize sum of radii derived from centers.
        We minimize -sum(radii).
        """
        r = _radii_from_centers(c)
        obj = -np.sum(r)
        # Gradient via finite differences would be slow; let's try to get analytic-ish
        # Actually, for SLSQP, we can provide the objective and let it handle gradients numerically,
        # or we can compute them. Given the soft-min, analytic gradient is complex.
        # Let's just return the objective and use SLSQP's numerical gradient, but limit iterations.
        return obj

    def _constraints(c):
        """
        Hard constraints for SLSQP:
        1. Centers inside [0,1]^2 (handled by bounds)
        2. No overlap: for all i<j, d_ij >= r_i + r_j
        3. Inside square: r_i <= x_i, etc.
        But since r is derived from c, we need to express constraints in terms of c.
        This is tricky. Instead, let's use a penalty approach within the objective,
        or reformulate.
        
        Alternative: Use the fact that if we define r_i = soft_min(constraints),
        then the "overlap" is automatically handled if we ensure the soft_min is tight.
        But SLSQP needs explicit constraints.
        
        Let's try a different approach: 
        Use L-BFGS-B on the 52 centers with a penalty for overlaps,
        where radii are computed from centers.
        """
        pass

    def _penalized_objective(c):
        """
        Minimize: -sum(r) + penalty * (overlap_violation + boundary_violation)
        """
        r = _radii_from_centers(c)
        x = c[:, 0]
        y = c[:, 1]
        
        # Check actual overlaps with the derived radii
        pts = c
        dist = cdist(pts, pts)
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        overlap = np.maximum(0, r_i + r_j - d_ij)
        overlap_pen = np.sum(overlap ** 2)
        
        # Boundary check
        b_pen = np.sum(np.maximum(0, r - x) ** 2)
        b_pen += np.sum(np.maximum(0, r - (1 - x)) ** 2)
        b_pen += np.sum(np.maximum(0, r - y) ** 2)
        b_pen += np.sum(np.maximum(0, r - (1 - y)) ** 2)
        
        total_pen = overlap_pen + b_pen
        return -np.sum(r) + 100.0 * total_pen

    def _run_optimization(c0, method='L-BFGS-B', maxiter=200):
        bounds = [(0.0, 1.0)] * (2 * N)
        if method == 'L-BFGS-B':
            res = minimize(_penalized_objective, c0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        elif method == 'SLSQP':
            # For SLSQP, we need constraints. Let's skip SLSQP for now and use L-BFGS-B.
            res = minimize(_penalized_objective, c0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10})
        return res

    def _extract_solution(c_opt):
        """Given optimized centers, compute final radii and return (N,3) array."""
        r = _radii_from_centers(c_opt)
        # Clip radii to be strictly within bounds
        x = c_opt[:, 0]
        y = c_opt[:, 1]
        # Ensure r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        wall_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_r)
        # Ensure no overlaps by reducing radii if needed
        pts = c_opt
        dist = cdist(pts, pts)
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        for _ in range(5):  # A few iterations to fix overlaps
            r_i = r[iu[0]]
            r_j = r[iu[1]]
            overlap = np.maximum(0, r_i + r_j - d_ij)
            if np.max(overlap) < 1e-9:
                break
            # Reduce both radii by half the overlap
            reduce_i = overlap / 2.0
            reduce_j = overlap / 2.0
            # Scatter back
            for k in range(len(iu[0])):
                i, j = iu[0][k], iu[1][k]
                r[i] -= reduce_i[k]
                r[j] -= reduce_j[k]
            r = np.clip(r, 1e-6, 0.5)
        
        return np.column_stack((x, y, r))

    # Generate initial center configurations
    inits = []
    
    # Init 1: 5x5 grid + 1 extra
    for i in range(5):
        for j in range(5):
            inits.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0])
    inits.append([0.05, 0.05])  # Extra near corner
    # Flatten each init to (2*N,)
    inits = [np.array(init).flatten() for init in inits]
    
    # Init 2: Random
    np.random.seed(42)
    inits.append(np.random.rand(N, 2).flatten())
    
    # Init 3: Hexagonal-like
    hex_inits = []
    row_height = 0.15
    for row in range(5):
        y_pos = 0.1 + row * row_height
        if row % 2 == 0:
            x_positions = [0.1 + i * 0.2 for i in range(5)]
        else:
            x_positions = [0.2 + i * 0.2 for i in range(5)]
        for x_pos in x_positions:
            if len(hex_inits) < N:
                hex_inits.append([x_pos, y_pos])
    if len(hex_inits) < N:
        hex_inits.append([0.5, 0.5])
    inits.append(np.array(hex_inits[:N]).flatten())
    
    # Init 4-6: More random restarts
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        inits.append(np.random.rand(N, 2).flatten())

    # Run optimization on each init
    for idx, c0 in enumerate(inits):
        if _elapsed() > timeout:
            break
        
        try:
            res = _run_optimization(c0, method='L-BFGS-B', maxiter=300)
            c_opt = res.x
            z_opt = _extract_solution(c_opt)
            
            # Validate
            if _valid(z_opt):
                sc = _score(z_opt)
                if sc > best_score:
                    best_score = sc
                    best_solution = z_opt
        except Exception:
            pass

    # If no valid solution found, fallback to grid
    if best_solution is None:
        cells = []
        for i in range(5):
            for j in range(5):
                cells.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0, 0.5 / 5.0 - 1e-3])
        cells.append([0.05, 0.05, 0.04])
        best_solution = np.array(cells[:N])

    return best_solution
