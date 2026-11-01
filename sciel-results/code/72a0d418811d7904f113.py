"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_radius(c, beta):
    """
    Compute a soft approximation of the minimum radius.
    Uses log-sum-exp to approximate the minimum of:
    - distances to boundaries
    - half pairwise distances
    """
    # Distances to boundaries
    # We want min(0.5 - |x|, 0.5 - |y|) for each point, which is min over 4 boundaries
    # Actually, min(0.5 - |c_i|) is the distance to the nearest boundary.
    # Let's construct the vector of all constraints:
    # 1. For each point i, distance to left, right, bottom, top boundaries.
    # 2. For each pair (i,j), half the distance between them.
    
    # Boundary terms: 0.5 - c[i,0], 0.5 + c[i,0], 0.5 - c[i,1], 0.5 + c[i,1]
    # Pair terms: 0.5 * ||c_i - c_j||
    
    # For efficiency, we can compute the soft min of the boundary distances and the soft min of the pair distances separately?
    # No, the min is over the union of all these constraints.
    # However, constructing the full vector of size 4N + N(N-1)/2 is expensive for N=200 (approx 40k terms).
    # Log-sum-exp on 40k terms is doable but might be slow if done repeatedly.
    
    # Alternative: Use a smaller set of active constraints or a different soft min.
    # Let's stick to the full set but optimize the calculation.
    
    # Boundary distances
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    # Pairwise distances
    # This is the bottleneck. pdist is fast in C.
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    # Combine all terms into one array
    # b_terms shape (4N), p_terms shape (N(N-1)/2)
    b_terms = np.concatenate([b_left, b_right, b_bottom, b_top])
    all_terms = np.concatenate([b_terms, p_terms])
    
    # Soft min: -1/beta * log(sum(exp(-beta * terms)))
    # To avoid overflow/underflow, we can use the log-sum-exp trick:
    # m = min(terms)
    # log_sum_exp(-beta * terms) = -beta * m + log(sum(exp(-beta * (terms - m))))
    # soft_min = -1/beta * (-beta * m + log(sum(exp(-beta * (terms - m)))))
    #           = m - 1/beta * log(sum(exp(-beta * (terms - m))))
    
    m = np.min(all_terms)
    # If m is very small, terms - m is small, exp is fine.
    # If beta is large, -beta * (terms - m) is <= 0, so exp is <= 1.
    # Sum of exps is <= N_terms.
    
    # We want to minimize the negative of the soft min, so we return -soft_min.
    # Or we can just return the soft min and minimize -soft_min.
    
    # For numerical stability with large beta:
    # If beta is very large, the term with the smallest value dominates.
    # exp(-beta * (terms - m)) will be 1 for the min terms and 0 for others.
    
    # Let's compute the log sum exp part
    # To avoid underflow for non-min terms when beta is large, we rely on the fact that they are > 0.
    # If terms - m > 1e-10, and beta=1000, exp(-100) is ~0.
    
    # Optimization: We don't need the exact sum if we are far from the minimum.
    # But for L-BFGS-B, we need a smooth gradient.
    
    # A cheaper approximation: only consider the k-smallest terms?
    # No, that introduces non-smoothness.
    
    # Let's use the full sum but be careful with performance.
    # N=200, terms ~ 40k. exp and sum on 40k floats is fast (~0.1ms).
    
    exp_terms = np.exp(-beta * (all_terms - m))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = m - (1.0 / beta) * log_sum
    
    return soft_min_val


def _objective(c_flat, beta):
    """Objective function for minimization: -soft_min_radius."""
    c = c_flat.reshape(-1, 2)
    # Ensure points are within bounds for the objective to be well-behaved?
    # L-BFGS-B handles bounds, but the objective should be smooth.
    # If a point goes outside, the boundary distance becomes negative.
    # The soft min will be negative. This is fine, it penalizes going outside.
    r = _soft_min_radius(c, beta)
    return -r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a few spacings
    for d in np.linspace(0.055, 0.075, 10):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_smooth(c0, max_time=30.0, beta=50.0):
    """Optimize using L-BFGS-B on the soft-min objective."""
    start_time = time.time()
    c0_flat = c0.flatten()
    
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Stage 1: Lower beta for wider exploration
    for b_val in [10.0, 20.0, 50.0, 100.0, 200.0]:
        if time.time() > start_time + max_time - 10:
            break
            
        # Create a closure to bind the current beta value
        def objective_wrapper(x, b=b_val):
            return _objective(x, b)
            
        try:
            res = minimize(
                objective_wrapper, 
                best_c.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            
            c_opt = res.x.reshape(-1, 2)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    return best_c, best_r


def _polish_active(c0, max_time=10.0):
    """
    Polish the solution by identifying active constraints and using SLSQP.
    """
    start_time = time.time()
    c = c0.copy()
    r_current = _compute_radius(c)
    
    # Identify active constraints:
    # 1. Find the minimum radius r.
    # 2. Find all pairs (i,j) such that dist(i,j)/2 <= r + eps.
    # 3. Find all points i and boundaries such that dist(i,boundary) <= r + eps.
    
    # We want to maximize r subject to:
    # dist(i,j) >= 2r
    # dist(i,boundary) >= r
    
    # This is a non-convex problem, but locally we can treat the active constraints as equalities.
    # SLSQP can handle inequality constraints.
    
    # Let's define the constraints for SLSQP.
    # Constraint: dist(i,j) - 2*r >= 0
    # Constraint: dist(i,boundary) - r >= 0
    
    # We need to pass r as a variable? No, we want to maximize r.
    # So we minimize -r.
    # Variables: c (2N vars) and r (1 var). Total 2N+1 vars.
    
    def objective(x):
        # x = [c_flat, r]
        c_flat = x[:2*N]
        r = x[2*N]
        c = c_flat.reshape(-1, 2)
        
        # Check validity: if any constraint is violated, return a large penalty
        # But SLSQP handles constraints via the `constraints` dict.
        # The objective is just -r.
        return -r
        
    def constraints(x):
        c_flat = x[:2*N]
        r = x[2*N]
        c = c_flat.reshape(-1, 2)
        
        cons = []
        
        # Boundary constraints
        # 0.5 - c[i,0] - r >= 0
        # 0.5 + c[i,0] - r >= 0
        # 0.5 - c[i,1] - r >= 0
        # 0.5 + c[i,1] - r >= 0
        
        # We can vectorize this? SLSQP expects a list of dicts.
        # It's better to provide a single function that returns an array of values for 'ineq'.
        
        # Let's use the 'ineq' type for all constraints.
        # We need to construct the array of all inequality constraints.
        
        # Boundary
        b_left = 0.5 - c[:, 0] - r
        b_right = 0.5 + c[:, 0] - r
        b_bottom = 0.5 - c[:, 1] - r
        b_top = 0.5 + c[:, 1] - r
        
        # Pairwise
        # This is expensive to compute for all pairs in every iteration.
        # N=200, pairs ~ 20k.
        # pdist is fast.
        d = pdist(c, metric='euclidean')
        p_terms = d - 2 * r
        
        # Combine
        all_ineq = np.concatenate([b_left, b_right, b_bottom, b_top, p_terms])
        
        # Return as a single 'ineq' constraint
        # SLSQP can handle a single 'ineq' constraint that returns an array.
        # Wait, scipy SLSQP 'ineq' expects a scalar or an array?
        # Documentation: "ineq : dict ... fun : function ... The function fun(x) must return a vector of length m ..."
        # Actually, for SLSQP, you can define multiple constraints.
        # If I return an array from one 'ineq' constraint, does it work?
        # Yes, "fun : function ... returns a vector".
        
        return {'type': 'ineq', 'fun': lambda x: all_ineq} # This lambda captures all_ineq which is wrong for gradient steps.
        # We need to define the function properly.
        
    # Defining the constraint function correctly for SLSQP is tricky with closures and performance.
    # Alternative: Use a simpler local search method like Nelder-Mead on the active set?
    # Or just use L-BFGS-B again with a very high beta?
    # The high beta L-BFGS-B effectively solves the active set problem.
    # So `_optimize_smooth` with high beta might be sufficient.
    
    # Let's try a simpler polish: Perturb the points slightly and re-optimize with L-BFGS-B high beta.
    # Or use `scipy.optimize.minimize` with method='Nelder-Mead' on the exact radius?
    # Nelder-Mead is derivative-free and robust, but slow in high dimensions (400D).
    
    # Let's stick to L-BFGS-B with increasing beta as the main optimizer.
    # The "polish" can just be another L-BFGS-B run with a very high beta (e.g., 1000).
    
    # So, I will remove the complex SLSQP attempt and just rely on the multi-stage L-BFGS-B.
    # The high beta stage acts as the polish.
    
    # For now, this function is a placeholder that does nothing or a very quick local step.
    # Let's implement a quick local step using finite differences on the exact radius?
    # No, that's slow.
    
    # Let's just return the input. The high-beta L-BFGS-B in `_optimize_smooth` should handle it.
    
    return c0, r_current


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (120s limit)
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Smooth optimization with L-BFGS-B
    if time.time() < deadline - 15:
        max_time_pgo = min(60.0, deadline - time.time() - 10)
        c_opt, r_opt = _optimize_smooth(c_hex, max_time=max_time_pgo)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # If we have time, try a second start from a slightly perturbed hexagonal grid
    # This helps if the first start got stuck in a local minimum.
    if time.time() < deadline - 25:
        np.random.seed(42)
        c_pert = best_c + np.random.normal(0, 0.005, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        max_time_pgo2 = min(40.0, deadline - time.time() - 10)
        c_opt2, r_opt2 = _optimize_smooth(c_pert, max_time=max_time_pgo2)
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
            
    # If we have more time, try a third start with different seed
    if time.time() < deadline - 35:
        np.random.seed(123)
        c_pert3 = best_c + np.random.normal(0, 0.01, size=(N, 2))
        c_pert3 = np.clip(c_pert3, -0.5, 0.5)
        
        max_time_pgo3 = min(30.0, deadline - time.time() - 10)
        c_opt3, r_opt3 = _optimize_smooth(c_pert3, max_time=max_time_pgo3)
        if r_opt3 > best_radius:
            best_radius = r_opt3
            best_c = c_opt3.copy()

    if best_c is None:
        best_c = _initialize_hexagonal()
    
    # Final projection to ensure strict validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
